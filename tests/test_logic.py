#!/usr/bin/env python3
"""Pure-logic tests for the CCIP grounding tools. No GPU, no models, no data.

Run from the repo root:  python3 -m unittest discover -s tests -v
Requires numpy (for the grounder/downsampler imports); imgutils is stubbed.
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ---- stub imgutils BEFORE importing anything that touches it ----
for _m in ("imgutils", "imgutils.detect", "imgutils.metrics"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["imgutils.detect"].detect_person = lambda *a, **k: []
sys.modules["imgutils.detect"].detect_heads = lambda *a, **k: []
for _fn in ("ccip_extract_feature", "ccip_batch_differences",
            "ccip_batch_extract_features", "ccip_default_threshold"):
    setattr(sys.modules["imgutils.metrics"], _fn, lambda *a, **k: None)

import numpy as np

import ccip_grounder
import ccip_downsample
from dbutil import open_v2, distinct_chars, paths_for_char


def make_v1_db(path, rows):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE images (name TEXT PRIMARY KEY, path TEXT, characters TEXT)")
    c.executemany("INSERT INTO images VALUES (?,?,?)", rows)
    c.commit()
    c.close()


class TestCropBestMatch(unittest.TestCase):
    def setUp(self):
        # 6 refs: 3 belong to char 0, 3 to char 1
        self.owner = np.array([0, 0, 0, 1, 1, 1])

    def test_picks_closer_character(self):
        row = np.array([0.9, 0.8, 0.85, 0.05, 0.06, 0.04])
        ci, score = ccip_grounder.crop_best_match(row, self.owner, 2, topk=1)
        self.assertEqual(ci, 1)
        self.assertAlmostEqual(score, 0.04)

    def test_topk_means_of_closest(self):
        row = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6])
        ci, s1 = ccip_grounder.crop_best_match(row, self.owner, 2, topk=1)
        self.assertEqual(ci, 0)
        self.assertAlmostEqual(s1, 0.1)
        ci2, s2 = ccip_grounder.crop_best_match(row, self.owner, 2, topk=2)
        self.assertAlmostEqual(s2, 0.15)  # mean(0.1, 0.2)

    def test_topk_clamped_to_available_refs(self):
        row = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        ci, s = ccip_grounder.crop_best_match(row, self.owner, 2, topk=99)
        self.assertEqual(ci, 0)
        self.assertAlmostEqual(s, float(np.mean([0.1, 0.2, 0.3])))


class TestParsePairs(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            ccip_grounder.parse_pairs(["a|b"]), [("a", "b")])

    def test_strips_whitespace_and_multiple(self):
        self.assertEqual(
            ccip_grounder.parse_pairs([" x | y ", " p|q "]),
            [("x", "y"), ("p", "q")])

    def test_splits_on_first_pipe_only(self):
        self.assertEqual(
            ccip_grounder.parse_pairs(["a|b|c"])[0][1], "b|c")

    def test_none_input(self):
        self.assertEqual(ccip_grounder.parse_pairs(None), [])

    def test_missing_pipe_exits(self):
        with self.assertRaises(SystemExit):
            ccip_grounder.parse_pairs(["no pipe here"])


class TestMaxMinSample(unittest.TestCase):
    def test_small_set_returned_whole(self):
        feats = np.random.RandomState(0).rand(5, 4)
        sel = ccip_downsample.max_min_sample(feats, 10)
        self.assertEqual(sorted(sel), list(range(5)))

    def test_selects_distinct_diverse_points(self):
        # two tight clusters far apart: sampling 2 must pick one from each
        rs = np.random.RandomState(42)
        cluster_a = rs.randn(20, 2) * 0.01 + [0, 0]
        cluster_b = rs.randn(20, 2) * 0.01 + [100, 100]
        feats = np.vstack([cluster_a, cluster_b])
        sel = ccip_downsample.max_min_sample(feats, 2)
        self.assertEqual(len(sel), 2)
        sides = {i < 20 for i in sel}
        self.assertEqual(sides, {True, False})

    def test_no_duplicates(self):
        rs = np.random.RandomState(7)
        feats = rs.rand(50, 8)
        sel = ccip_downsample.max_min_sample(feats, 20)
        self.assertEqual(len(sel), len(set(sel)))
        self.assertEqual(len(sel), 20)


class TestPreferSolo(unittest.TestCase):
    """--prefer-solo: solo pool fills the target first, group shots backfill."""

    def _feats_and_tags(self, n_solo, n_group, seed=0):
        rs = np.random.RandomState(seed)
        solo = rs.rand(n_solo, 4) * 0.01 + [0, 0]
        group = rs.rand(n_group, 4) * 0.01 + [100, 100]
        feats = np.vstack([solo, group])
        n_tags = np.array([1] * n_solo + [2] * n_group)
        return feats, n_tags

    def test_solo_excludes_group_when_solo_fills_target(self):
        feats, n_tags = self._feats_and_tags(150, 100)
        sel = ccip_downsample.select_prefer_solo(feats, n_tags, 100)
        self.assertEqual(len(sel), 100)
        self.assertTrue(all(n_tags[i] == 1 for i in sel))

    def test_group_backfills_solo_shortfall(self):
        feats, n_tags = self._feats_and_tags(30, 200)
        sel = ccip_downsample.select_prefer_solo(feats, n_tags, 100)
        self.assertEqual(len(sel), 100)
        self.assertEqual(sum(1 for i in sel if n_tags[i] == 1), 30)

    def test_small_pools_returned_whole(self):
        feats, n_tags = self._feats_and_tags(10, 8)
        sel = ccip_downsample.select_prefer_solo(feats, n_tags, 100)
        self.assertEqual(sorted(sel), list(range(18)))

    def test_no_duplicates_and_capped(self):
        feats, n_tags = self._feats_and_tags(90, 90, seed=7)
        sel = ccip_downsample.select_prefer_solo(feats, n_tags, 100)
        self.assertEqual(len(sel), len(set(sel)))
        self.assertLessEqual(len(sel), 100)

    def test_group_pool_internally_diverse(self):
        # backfilled group picks must Max-Min across the group cluster, not
        # take the first N: seed two sub-clusters inside the group pool.
        rs = np.random.RandomState(3)
        solo = rs.rand(5, 2) * 0.01 + [0, 0]
        gA = rs.rand(50, 2) * 0.01 + [100, 100]
        gB = rs.rand(50, 2) * 0.01 + [200, 200]
        feats = np.vstack([solo, gA, gB])
        n_tags = np.array([1] * 5 + [2] * 100)
        sel = ccip_downsample.select_prefer_solo(feats, n_tags, 55)
        grp = [i - 5 for i in sel if n_tags[i] > 1]
        in_A = sum(1 for i in grp if i < 50)
        in_B = sum(1 for i in grp if i >= 50)
        self.assertGreater(in_A, 0)
        self.assertGreater(in_B, 0)


class TestDBManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sql", delete=False)
        self.tmp.close()
        self.db = ccip_grounder.DBManager(self.tmp.name)

    def tearDown(self):
        self.db.close()
        for suf in ("", "-wal", "-shm"):
            p = self.tmp.name + suf
            if os.path.exists(p):
                os.remove(p)

    def test_register_dedupes_by_path(self):
        self.assertEqual(self.db.register_images(["/a/1.png", "/b/1.png"]), 2)
        self.assertEqual(self.db.register_images(["/a/1.png"]), 0)

    def test_same_basename_different_paths_coexist(self):
        # the v1 basename-PK bug: these used to collide silently
        self.db.register_images(["/dir1/x.png", "/dir2/x.png"])
        self.db.add_character_tag("/dir1/x.png", "sara")
        self.db.add_character_tag("/dir2/x.png", "zenith")
        self.assertEqual(paths_for_char(self.db.conn, "sara"), ["/dir1/x.png"])
        self.assertEqual(paths_for_char(self.db.conn, "zenith"), ["/dir2/x.png"])

    def test_tag_idempotent_and_purge(self):
        self.db.register_images(["/a.png", "/b.png"])
        self.db.add_character_tag("/a.png", "eris")
        self.db.add_character_tag("/a.png", "eris")
        self.db.add_character_tag("/b.png", "eris")
        self.assertEqual(self.db.purge_character("eris"), 2)
        self.assertEqual(paths_for_char(self.db.conn, "eris"), [])

    def test_prune_missing_keeps_tags_consistent(self):
        self.db.register_images(["/definitely/missing.png"])
        self.db.add_character_tag("/definitely/missing.png", "roxy")
        self.assertEqual(self.db.prune_missing(), 1)
        self.assertEqual(self.db.get_all_images(), [])
        # no orphan tags left behind
        n = self.db.conn.execute("SELECT COUNT(*) FROM image_tags").fetchone()[0]
        self.assertEqual(n, 0)

    def test_legacy_detection(self):
        self.assertFalse(self.db.detect_legacy())
        tmp2 = tempfile.NamedTemporaryFile(suffix=".sql", delete=False)
        tmp2.close()
        make_v1_db(tmp2.name, [("a.png", "/a.png", "x")])
        db2 = ccip_grounder.DBManager(tmp2.name)
        self.assertTrue(db2.detect_legacy())
        db2.close()
        os.remove(tmp2.name)


class TestDbutil(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sql", delete=False)
        self.tmp.close()
        make_v1_db(self.tmp.name, [
            ("a.png", "/x/a.png", "rudeus"),
            ("b.png", "/x/b.png", "rudeus_child,sara"),
            ("c.png", "/x/c.png", ""),
        ])
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "migrate_v2.py"),
             "--db", self.tmp.name],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def tearDown(self):
        for suf in ("", "-wal", "-shm"):
            p = self.tmp.name + suf
            if os.path.exists(p):
                os.remove(p)

    def test_migration_preserves_counts(self):
        conn = open_v2(self.tmp.name)
        self.assertEqual(sorted(distinct_chars(conn)),
                         ["rudeus", "rudeus_child", "sara"])
        self.assertEqual(len(paths_for_char(conn, "rudeus")), 1)
        conn.close()

    def test_exact_match_no_substring_bleed(self):
        conn = open_v2(self.tmp.name)
        # LIKE '%rudeus%' would have matched rudeus_child too
        self.assertEqual(paths_for_char(conn, "rudeus"), ["/x/a.png"])
        self.assertEqual(paths_for_char(conn, "rudeus_child"), ["/x/b.png"])
        conn.close()

    def test_legacy_db_rejected(self):
        tmp2 = tempfile.NamedTemporaryFile(suffix=".sql", delete=False)
        tmp2.close()
        make_v1_db(tmp2.name, [("a.png", "/a.png", "x")])
        with self.assertRaises(SystemExit):
            open_v2(tmp2.name)
        os.remove(tmp2.name)

    def test_migration_is_idempotent(self):
        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "migrate_v2.py"),
             "--db", self.tmp.name],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("already uses the v2 schema", r.stdout)


class TestHardCap(unittest.TestCase):
    def _make_v2_db(self, path, tags_by_path):
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE images (path TEXT PRIMARY KEY, name TEXT)")
        c.execute("CREATE TABLE image_tags (path TEXT NOT NULL, char TEXT NOT NULL,"
                  " PRIMARY KEY (path, char))")
        for p, tags in tags_by_path.items():
            c.execute("INSERT INTO images VALUES (?, ?)", (p, os.path.basename(p)))
            for t in tags:
                c.execute("INSERT INTO image_tags VALUES (?, ?)", (p, t))
        c.commit()
        c.close()

    def test_cap_enforced_and_healthy_chars_survive(self):
        import ccip_hard_cap
        tmp = tempfile.NamedTemporaryFile(suffix=".sql", delete=False)
        tmp.close()
        tags = {f"/x/big_{i}.png": ["big"] for i in range(20)}
        tags.update({f"/x/small_{i}.png": ["small"] for i in range(5)})
        self._make_v2_db(tmp.name, tags)

        ccip_hard_cap.hard_cap_database(tmp.name, limit=10, seed=123)

        conn = open_v2(tmp.name)
        counts = dict(conn.execute("SELECT char, COUNT(*) FROM image_tags GROUP BY char"))
        conn.close()
        self.assertLessEqual(counts["big"], 10)
        # single-char pruning must not starve the healthy character
        self.assertEqual(counts["small"], 5)
        os.remove(tmp.name)

    def test_noop_when_under_limit(self):
        import ccip_hard_cap
        tmp = tempfile.NamedTemporaryFile(suffix=".sql", delete=False)
        tmp.close()
        self._make_v2_db(tmp.name, {f"/x/{i}.png": ["c"] for i in range(3)})
        ccip_hard_cap.hard_cap_database(tmp.name, limit=100, seed=1)
        conn = open_v2(tmp.name)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM images").fetchone()[0], 3)
        conn.close()
        os.remove(tmp.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)

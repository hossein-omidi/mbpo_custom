import unittest

from SAC_workspace.src.training.experiment_protocol import build_scenario_protocol_from_rows


class ScenarioProtocolTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for year in range(2018, 2025):
            for month in range(1, 13):
                for day in range(1, 6):
                    rows.append({
                        "scenario_id": f"{year:04d}-{month:02d}-{day:02d}",
                        "source_year": year, "month": month, "day": day,
                        "probability": 1.0,
                    })
        return rows

    def test_partition_is_disjoint_exhaustive_and_month_balanced(self):
        rows = self.make_rows()
        protocol = build_scenario_protocol_from_rows(
            rows, manifest_sha256="test-manifest", split_seed=2026,
            validation_days_per_month=1, test_days_per_month=2,
            preferred_validation_ids=["2018-02-03", "2020-08-04"],
        )
        train = set(protocol["train_ids"])
        validation = set(protocol["validation_ids"])
        test = set(protocol["test_ids"])
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test,
                         {r["scenario_id"] for r in rows})
        self.assertEqual(len(validation), 12)
        self.assertEqual(len(test), 24)
        self.assertIn("2018-02-03", validation)
        self.assertIn("2020-08-04", validation)
        for month in range(1, 13):
            self.assertEqual(protocol["month_counts"][str(month)]["validation"], 1)
            self.assertEqual(protocol["month_counts"][str(month)]["test"], 2)
        self.assertAlmostEqual(sum(protocol["train_probabilities"]), 1.0)

    def test_partition_is_deterministic(self):
        rows = self.make_rows()
        a = build_scenario_protocol_from_rows(rows, manifest_sha256="same", split_seed=7)
        b = build_scenario_protocol_from_rows(rows, manifest_sha256="same", split_seed=7)
        self.assertEqual(a, b)
        c = build_scenario_protocol_from_rows(rows, manifest_sha256="same", split_seed=8)
        self.assertNotEqual(a["protocol_sha256"], c["protocol_sha256"])


if __name__ == "__main__":
    unittest.main()

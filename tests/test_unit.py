import ctypes
import tempfile
import unittest
from pathlib import Path

from runner import abi, sycl
from runner.challenge import resolve
from runner.result import GradeResult
from service.registry import Registry
from service.validate import validate_submission


class FakeChallenge:
    def __init__(self, signature):
        self._signature = signature

    def get_solve_signature(self):
        return self._signature


class FakeTensor:
    def __init__(self, ptr):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr


class SignatureTests(unittest.TestCase):
    def test_pointer_becomes_tensor_arg(self):
        specs = abi.parse_signature(
            FakeChallenge(
                {
                    "A": (ctypes.POINTER(ctypes.c_float), "in"),
                    "out": (ctypes.POINTER(ctypes.c_int), "out"),
                }
            )
        )
        self.assertEqual([spec.kind for spec in specs], ["tensor", "tensor"])
        self.assertEqual(abi.output_names(specs), ["out"])

    def test_scalars(self):
        specs = abi.parse_signature(
            FakeChallenge(
                {
                    "N": (ctypes.c_int, "in"),
                    "alpha": (ctypes.c_float, "in"),
                    "size": (ctypes.c_size_t, "in"),
                }
            )
        )
        self.assertEqual([spec.kind for spec in specs], ["scalar"] * 3)
        args = abi.raw_args(specs, {"N": 4, "alpha": 0.5, "size": 8})
        self.assertEqual(args, [4, 0.5, 8])

    def test_tensor_args_use_data_ptr(self):
        specs = abi.parse_signature(FakeChallenge({"A": (ctypes.POINTER(ctypes.c_float), "in")}))
        args = abi.raw_args(specs, {"A": FakeTensor(0x1234)})
        self.assertEqual(args[0].value, 0x1234)

    def test_rejects_unknown_type(self):
        with self.assertRaises(TypeError):
            abi.parse_signature(FakeChallenge({"x": (object, "in")}))

    def test_rejects_unknown_direction(self):
        with self.assertRaises(ValueError):
            abi.parse_signature(FakeChallenge({"x": (ctypes.c_int, "sideways")}))


class SyclCommandTests(unittest.TestCase):
    def test_default_flags(self):
        command = sycl.build_command(Path("a.cpp"), Path("a.so"), "/usr/bin/icpx")
        self.assertEqual(command[:4], ["/usr/bin/icpx", "-fsycl", "-O2", "-fPIC"])
        self.assertIn("-shared", command)
        self.assertEqual(command[-3:], ["-o", "a.so", "a.cpp"])

    def test_extra_flags(self):
        command = sycl.build_command(Path("a.cpp"), Path("a.so"), "icpx", ["-g"])
        self.assertIn("-g", command)


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "challenges"
        for rel in ("easy/1_vector_add", "medium/5_softmax", "hard/39_fast_fourier_transform"):
            path = root / rel
            path.mkdir(parents=True)
            (path / "challenge.py").write_text("class Challenge: pass\n")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve_accepts_repo_root(self):
        self.assertEqual(resolve(self.root.parent, "5").name, "5_softmax")

    def test_resolve_by_id(self):
        self.assertEqual(resolve(self.root, "5").name, "5_softmax")

    def test_resolve_by_slug(self):
        self.assertEqual(resolve(self.root, "vector-add").name, "1_vector_add")

    def test_resolve_by_relative_path(self):
        self.assertEqual(resolve(self.root, "medium/5_softmax").name, "5_softmax")

    def test_resolve_by_directory(self):
        path = self.root / "easy/1_vector_add"
        self.assertEqual(resolve(self.root, str(path)), path.resolve())

    def test_unknown_slug_exits(self):
        with self.assertRaises(SystemExit):
            resolve(self.root, "does-not-exist")


class ResultTests(unittest.TestCase):
    def test_summary_counts(self):
        result = GradeResult(status="success", challenge="easy/1_vector_add")
        self.assertEqual(result.to_dict()["functional"], {"passed": 0, "total": 0, "cases": []})
        self.assertIsNone(result.to_dict()["performance"])


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "challenges"
        path = root / "easy" / "1_vector_add"
        (path / "starter").mkdir(parents=True)
        (path / "challenge.py").write_text(
            'class Challenge:\n    name = "Vector Addition"\n    access_tier = "free"\n'
        )
        (path / "challenge.html").write_text("<p>add</p>")
        (path / "starter" / "starter.sycl.cpp").write_text("// sycl\n")
        self.root = root
        self.registry = Registry(root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_entry_metadata(self):
        entry = self.registry.get(1)
        self.assertEqual(entry.title, "Vector Addition")
        self.assertEqual(entry.access_tier, "free")
        self.assertIn("sycl", entry.starters)

    def test_exact_match_required(self):
        entry = self.registry.get(1)
        self.assertTrue(self.registry.matches(entry, entry.source))
        self.assertFalse(self.registry.matches(entry, entry.source + "\n# tampered\n"))

    def test_resolution_by_title_when_source_differs(self):
        entry = self.registry.get(1)
        self.assertEqual(self.registry.find_by_source(entry.source + "\n# x\n").id, 1)


class ServiceValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "challenges"
        path = root / "easy" / "1_vector_add"
        (path / "starter").mkdir(parents=True)
        self.source = 'class Challenge:\n    name = "Vector Addition"\n    access_tier = "free"\n'
        (path / "challenge.py").write_text(self.source)
        (path / "challenge.html").write_text("<p>add</p>")
        (path / "starter" / "starter.sycl.cpp").write_text("// sycl\n")
        self.registry = Registry(root)

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, **overrides):
        submission = {
            "files": [{"name": "solution.cpp", "content": "// kernel"}],
            "language": "sycl",
            "accelerator": "B70",
            "challengeCode": self.source,
        }
        submission.update(overrides.pop("submission", {}))
        payload = {"action": "run", "token": "test-key", "submission": submission}
        payload.update(overrides)
        return payload

    def test_accepts_valid_submission(self):
        request, problem = validate_submission(self.payload(), "test-key", self.registry)
        self.assertIsNone(problem)
        self.assertEqual(request["challenge"], "easy/1_vector_add")
        self.assertEqual(request["file_name"], "solution.cpp")

    def test_rejects_bad_token(self):
        _, problem = validate_submission(self.payload(token="nope"), "test-key", self.registry)
        self.assertEqual(problem["output"], "invalid API token")

    def test_rejects_language(self):
        _, problem = validate_submission(
            self.payload(submission={"language": "cuda"}), "test-key", self.registry
        )
        self.assertIn("unsupported language", problem["output"])

    def test_rejects_missing_source(self):
        _, problem = validate_submission(
            self.payload(submission={"files": [{"name": "a.py", "content": ""}]}),
            "test-key",
            self.registry,
        )
        self.assertIn("no .cpp file", problem["output"])

    def test_rejects_tampered_oracle(self):
        _, problem = validate_submission(
            self.payload(submission={"challengeCode": self.source + "\n# tampered\n"}),
            "test-key",
            self.registry,
        )
        self.assertIn("differs from the first-party challenge", problem["output"])

    def test_rejects_unknown_oracle(self):
        _, problem = validate_submission(
            self.payload(submission={"challengeCode": "class Nothing: pass\n"}),
            "test-key",
            self.registry,
        )
        self.assertIn("does not match any first-party challenge", problem["output"])


if __name__ == "__main__":
    unittest.main()

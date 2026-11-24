"""
Unit tests for ModelInspector and AutoSelector capability-driven routing.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from workspace_core.hardware import HardwareProfile, GPUInfo
from workspace_core.model_inspector import GGUFVariant, InspectionResult, ModelInspector
from workspace_core.auto_selector import AutoSelector


class TestModelInspectorAndAutoSelector(unittest.TestCase):
    def setUp(self):
        self.modern_gpu = GPUInfo(
            index=0,
            name="NVIDIA L4",
            sm_name="sm_89",
            compute_capability=(8, 9),
            total_vram_mb=24000.0,
            free_vram_mb=22000.0,
            used_vram_mb=2000.0,
            supports_flash_attn=True,
            supports_kv_quant=True,
            supports_vllm=True,
            supports_bf16=True,
            supports_fp8=True,
        )
        self.modern_hw = HardwareProfile(
            has_gpu=True,
            gpus=[self.modern_gpu],
            primary_gpu=self.modern_gpu,
            ram_total_gb=32.0,
            ram_available_gb=28.0,
            ram_used_gb=4.0,
            ram_percent=12.5,
            ssd_total_gb=100.0,
            ssd_free_gb=80.0,
            cpu_cores=8,
            cuda_version="12.2",
            cuda_major=12,
            driver_version="535.104.05",
            python_version="3.10.12",
            python_executable="/usr/bin/python3",
        )

        self.t4_gpu = GPUInfo(
            index=0,
            name="Tesla T4",
            sm_name="sm_75",
            compute_capability=(7, 5),
            total_vram_mb=15109.0,
            free_vram_mb=14000.0,
            used_vram_mb=1109.0,
            supports_flash_attn=True,
            supports_kv_quant=True,
            supports_vllm=False,
        )
        self.t4_hw = HardwareProfile(
            has_gpu=True,
            gpus=[self.t4_gpu],
            primary_gpu=self.t4_gpu,
            ram_total_gb=12.0,
            ram_available_gb=10.0,
            ram_used_gb=2.0,
            ram_percent=16.6,
            ssd_total_gb=100.0,
            ssd_free_gb=70.0,
            cpu_cores=2,
            cuda_version="12.2",
            cuda_major=12,
            driver_version="535.104.05",
            python_version="3.10.12",
            python_executable="/usr/bin/python3",
        )

    def test_standard_gguf_selects_llama_cpp(self):
        """Standard GGUF model should select llama.cpp engine and converter."""
        insp = InspectionResult(
            source_type="hf_repo",
            raw_source="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            model_id="Qwen2.5-0.5B-Instruct-GGUF",
            model_type="llm-gguf",
            quant_format="gguf_standard",
            gguf_variants=[
                GGUFVariant(
                    name="qwen2.5-0.5b-instruct-q4_k_m.gguf",
                    quant="Q4_K_M",
                    is_split=False,
                    shard_files=["qwen2.5-0.5b-instruct-q4_k_m.gguf"],
                    total_size_bytes=350 * 1024 * 1024,
                    total_size_gb=0.35,
                    is_ik_quant=False,
                )
            ],
        )
        selector = AutoSelector(hardware=self.t4_hw)
        rec = selector.recommend(insp)

        self.assertEqual(rec.recommended_engine, "llama.cpp")
        self.assertEqual(rec.recommended_converter, "llama.cpp")
        self.assertEqual(rec.recommended_quant, "Q4_K_M")
        self.assertIn("llama.cpp", rec.decision_summary)

    def test_ik_quant_or_moe_selects_ik_llama(self):
        """Non-uniform IQ quants (e.g. IQ4_KS) or oversized MoE models should select ik_llama."""
        insp_ik = InspectionResult(
            source_type="local",
            raw_source="/content/model-IQ4_KS.gguf",
            model_id="model-IQ4_KS",
            model_type="llm-gguf",
            is_ik_quant=True,
            quant_format="gguf_ik",
            gguf_variants=[
                GGUFVariant(
                    name="model-IQ4_KS.gguf",
                    quant="IQ4_KS",
                    is_split=False,
                    shard_files=["model-IQ4_KS.gguf"],
                    total_size_bytes=4 * 1024 * 1024 * 1024,
                    total_size_gb=4.0,
                    is_ik_quant=True,
                )
            ],
        )
        selector_t4 = AutoSelector(hardware=self.t4_hw)
        rec_ik = selector_t4.recommend(insp_ik)
        self.assertEqual(rec_ik.recommended_engine, "ik_llama")
        self.assertEqual(rec_ik.recommended_converter, "ik_llama")

        insp_moe = InspectionResult(
            source_type="hf_repo",
            raw_source="mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF",
            model_id="Mixtral-8x7B-Instruct-GGUF",
            model_type="llm-gguf",
            is_moe=True,
            quant_format="gguf_standard",
            architecture_name="MixtralForCausalLM",
            gguf_variants=[
                GGUFVariant(
                    name="mixtral-8x7b-q4_k_m.gguf",
                    quant="Q4_K_M",
                    is_split=False,
                    shard_files=["mixtral-8x7b-q4_k_m.gguf"],
                    total_size_bytes=26 * 1024 * 1024 * 1024,
                    total_size_gb=26.0,
                    is_ik_quant=False,
                )
            ],
        )
        rec_moe = selector_t4.recommend(insp_moe)
        self.assertEqual(rec_moe.recommended_engine, "ik_llama")
        self.assertIn("override_tensor", rec_moe.suggested_extra_args)

    def test_lora_or_unsloth_selects_unsloth(self):
        """LoRA adapter repo or Unsloth repo should select unsloth engine and converter."""
        insp_lora = InspectionResult(
            source_type="hf_repo",
            raw_source="user/llama3-lora-adapter",
            model_id="llama3-lora-adapter",
            model_type="llm-safetensors",
            is_lora=True,
            quant_format="unquantized_fp16",
        )
        selector = AutoSelector(hardware=self.t4_hw)
        rec_lora = selector.recommend(insp_lora)
        self.assertEqual(rec_lora.recommended_engine, "unsloth")
        self.assertEqual(rec_lora.recommended_converter, "unsloth")

    def test_awq_fp8_hardware_routing_vllm_vs_transformers(self):
        """AWQ or FP8 on modern GPU (L4 sm_89+) selects vLLM; on T4 (sm_75) selects transformers."""
        insp_awq = InspectionResult(
            source_type="hf_repo",
            raw_source="TheBloke/Llama-2-7B-AWQ",
            model_id="Llama-2-7B-AWQ",
            model_type="llm-safetensors",
            quant_format="awq",
        )

        selector_l4 = AutoSelector(hardware=self.modern_hw)
        rec_l4 = selector_l4.recommend(insp_awq)
        self.assertEqual(rec_l4.recommended_engine, "vllm")
        self.assertEqual(rec_l4.recommended_converter, "autoawq")

        selector_t4 = AutoSelector(hardware=self.t4_hw)
        rec_t4 = selector_t4.recommend(insp_awq)
        self.assertEqual(rec_t4.recommended_engine, "transformers")
        self.assertEqual(rec_t4.recommended_converter, "autoawq")

    def test_exl2_selects_exllamav2(self):
        """EXL2 models should select exllamav2 engine and converter."""
        insp_exl2 = InspectionResult(
            source_type="hf_repo",
            raw_source="turboderp/Llama2-7B-exl2",
            model_id="Llama2-7B-exl2",
            model_type="llm-safetensors",
            quant_format="exl2",
        )
        selector = AutoSelector(hardware=self.t4_hw)
        rec = selector.recommend(insp_exl2)
        self.assertEqual(rec.recommended_engine, "exllamav2")
        self.assertEqual(rec.recommended_converter, "exllamav2")

    def test_whisper_selects_ct2_faster_whisper(self):
        """Whisper audio models should select ct2 converter and faster-whisper engine."""
        insp_audio = InspectionResult(
            source_type="hf_repo",
            raw_source="openai/whisper-large-v3",
            model_id="whisper-large-v3",
            model_type="audio",
        )
        selector = AutoSelector(hardware=self.t4_hw)
        rec = selector.recommend(insp_audio)
        self.assertEqual(rec.recommended_engine, "faster-whisper")
        self.assertEqual(rec.recommended_converter, "ct2")

    def test_manual_overrides_strictly_respected(self):
        """User manual overrides must be strictly honored by AutoSelector."""
        insp = InspectionResult(
            source_type="hf_repo",
            raw_source="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            model_id="Qwen2.5-0.5B-Instruct-GGUF",
            model_type="llm-gguf",
            quant_format="gguf_standard",
            gguf_variants=[
                GGUFVariant(
                    name="qwen2.5-0.5b-instruct-q4_k_m.gguf",
                    quant="Q4_K_M",
                    is_split=False,
                    shard_files=["qwen2.5-0.5b-instruct-q4_k_m.gguf"],
                    total_size_bytes=350 * 1024 * 1024,
                    total_size_gb=0.35,
                    is_ik_quant=False,
                )
            ],
        )
        selector = AutoSelector(hardware=self.t4_hw)
        rec = selector.recommend(
            inspection=insp,
            requested_ctx=8192,
            override_engine="ik_llama",
            override_quantizer="ik_llama",
            override_quant="Q4_K_M",
            override_n_gpu_layers=20,
            override_flash_attn=False,
            override_cache_type_k="q4_0",
            override_cache_type_v="q4_0",
        )

        self.assertEqual(rec.recommended_engine, "ik_llama")
        self.assertEqual(rec.recommended_converter, "ik_llama")
        self.assertEqual(rec.recommended_quant, "Q4_K_M")
        self.assertEqual(rec.recommended_ctx, 8192)
        self.assertEqual(rec.recommended_n_gpu_layers, 20)
        self.assertFalse(rec.flash_attn)
        self.assertEqual(rec.cache_type_k, "q4_0")
        self.assertEqual(rec.cache_type_v, "q4_0")

    def test_manual_quant_not_in_repo_triggers_requant(self):
        """When user manually overrides target_quant to a format not in repo, trigger re-quantization."""
        insp = InspectionResult(
            source_type="hf_repo",
            raw_source="test/repo-gguf",
            model_id="repo-gguf",
            model_type="llm-gguf",
            quant_format="gguf_standard",
            gguf_variants=[
                GGUFVariant(
                    name="model-q8_0.gguf",
                    quant="Q8_0",
                    is_split=False,
                    shard_files=["model-q8_0.gguf"],
                    total_size_bytes=8 * 1024 * 1024 * 1024,
                    total_size_gb=8.0,
                    is_ik_quant=False,
                )
            ],
        )
        selector = AutoSelector(hardware=self.t4_hw)
        rec = selector.recommend(inspection=insp, override_quant="Q4_K_M")
        self.assertTrue(rec.needs_requant)
        self.assertEqual(rec.requant_target_quant, "Q4_K_M")
        self.assertEqual(rec.recommended_quant, "Q4_K_M")

    def test_vllm_repo_name_does_not_false_trigger_vlm(self):
        """Repo names containing 'vllm' must NOT be misclassified as Vision-Language (VLM)."""
        mtype, backend = ModelInspector._classify_architecture(
            repo_id="org/llama-3-8b-vllm-fp8",
            pipeline_tag="text-generation",
            tags=[],
            has_gguf=False,
            has_safetensors=True,
            has_mmproj=False,
            file_names=["model.safetensors", "config.json"],
        )
        self.assertEqual(mtype, "llm-safetensors")
        self.assertEqual(backend, "vllm")

    def test_convert_and_serve_safetensors_on_llama_cpp(self):
        """Safetensors targeted for llama.cpp or convert_and_serve should set needs_convert_to_gguf=True."""
        insp = InspectionResult(
            source_type="hf_repo",
            raw_source="Qwen/Qwen2.5-3B-Instruct",
            model_id="Qwen2.5-3B-Instruct",
            model_type="llm-safetensors",
            quant_format="unquantized_fp16",
            total_repo_size_bytes=6 * 1024 * 1024 * 1024,
        )
        selector = AutoSelector(hardware=self.t4_hw)
        rec = selector.recommend(insp, task="convert_and_serve", override_engine="llama.cpp")
        self.assertTrue(rec.needs_convert_to_gguf)
        self.assertEqual(rec.recommended_engine, "llama.cpp")
        self.assertEqual(rec.recommended_converter, "llama.cpp")

    def test_local_split_gguf_and_sibling_mmproj_discovery(self):
        """Inspecting shard 1 of a split local GGUF should auto-discover all shards and sibling mmproj."""
        tmp = Path(tempfile.mkdtemp(prefix="test_split_"))
        try:
            s1 = tmp / "VLM-Q4_K_M-00001-of-00002.gguf"
            s2 = tmp / "VLM-Q4_K_M-00002-of-00002.gguf"
            mm = tmp / "mmproj-model-f16.gguf"
            s1.write_bytes(b"shard1")
            s2.write_bytes(b"shard2")
            mm.write_bytes(b"mmproj")

            insp = ModelInspector().inspect(str(s1), input_mode="local_path")
            self.assertEqual(insp.model_type, "vlm-gguf")
            self.assertTrue(insp.is_vlm)
            self.assertEqual(len(insp.gguf_variants), 1)
            self.assertTrue(insp.gguf_variants[0].is_split)
            self.assertEqual(len(insp.gguf_variants[0].shard_files), 2)
            self.assertIsNotNone(insp.gguf_variants[0].mmproj_file)
        finally:
            shutil.rmtree(str(tmp))

    def test_strategies_and_auto_context(self):
        """AutoSelector should support 'auto' context and distinct strategies."""
        insp = InspectionResult(
            source_type="hf_repo",
            raw_source="test/model-gguf",
            model_id="model-gguf",
            model_type="llm-gguf",
            quant_format="gguf_standard",
            max_context_length=16384,
            gguf_variants=[
                GGUFVariant("m-Q8_0", "Q8_0", False, ["m-Q8_0.gguf"], 8 * 1024**3, 8.0),
                GGUFVariant("m-Q4_K_M", "Q4_K_M", False, ["m-Q4_K_M.gguf"], 4 * 1024**3, 4.0),
            ],
        )
        selector = AutoSelector(hardware=self.modern_hw)
        rec_speed = selector.recommend(insp, requested_ctx="auto", strategy="max_speed")
        rec_ctx = selector.recommend(insp, requested_ctx="auto", strategy="max_context")
        self.assertEqual(rec_speed.recommended_quant, "Q4_K_M")
        self.assertEqual(rec_ctx.recommended_ctx, 16384)
        self.assertEqual(rec_ctx.cache_type_k, "q4_0")


if __name__ == "__main__":
    unittest.main()

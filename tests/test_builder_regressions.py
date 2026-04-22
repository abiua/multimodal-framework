import unittest
from types import SimpleNamespace

from models.builder import ModelBuilder, StageFusionAdapter


class TestBuilderRegressions(unittest.TestCase):
    def test_build_model_should_pass_use_staged_forward_and_fusion_stages(self):
        config = SimpleNamespace(
            data=SimpleNamespace(modalities=["image"]),
            model=SimpleNamespace(
                backbones={
                    "image": SimpleNamespace(
                        type="resnet18_stageable",
                        feature_dim=32,
                        pretrained=False,
                        freeze=False,
                        extra_params={},
                    )
                },
                dropout_rate=0.1,
                fusion_hidden_dim=64,
                fusion_type="concat",
                classifier_hidden_dims=[],
                use_staged_forward=True,
                fusion_stages=[0, 3],
            ),
            classes=SimpleNamespace(num_classes=5),
        )

        model = ModelBuilder.build_model(config)

        self.assertTrue(model.use_staged_forward)
        self.assertEqual(model.fusion_stages, {0, 3})

    @unittest.expectedFailure
    def test_stage_fusion_adapter_should_be_constructible(self):
        adapter = StageFusionAdapter(
            common_dim=16,
            modality_channels={
                "image": 32,
                "wave": 64,
            },
        )

        self.assertIn("image", adapter.to_common)
        self.assertIn("wave", adapter.to_common)


if __name__ == "__main__":
    unittest.main()
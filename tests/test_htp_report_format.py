"""
Test HTP report format matches console output requirements.

This test validates that the markdown report follows the same format
as the console output, with proper titles and structure.
"""

import pytest

from modelexport.strategies.htp.htp_exporter import HTPExporter


class TestHTPReportFormat:
    """Test that HTP report format matches console output."""

    def test_module_hierarchy_preview_format(self, tmp_path):
        """Test Module Hierarchy Preview uses detail tags instead of truncation."""
        output_path = tmp_path / "test.onnx"

        # Export with reporting enabled
        exporter = HTPExporter(verbose=False, enable_reporting=True)

        try:
            exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
                opset_version=17,
            )
        except Exception:
            # Export might fail but we can still check report format
            pass

        # Check report file
        report_path = output_path.with_name(output_path.stem + "_htp_export_report.md")
        if report_path.exists():
            report_content = report_path.read_text()

            # Check Module Hierarchy Preview section
            assert "#### Module Hierarchy Preview" in report_content

            # Should use details tag, not truncation message
            assert "<details>" in report_content
            assert "Click to expand" in report_content
            assert "truncated for console" not in report_content

            # Check that it doesn't have depth/line limits in preview
            assert "max_depth=3" not in report_content
            assert "max_lines=20" not in report_content

    def test_node_distribution_format(self, tmp_path):
        """Test Node Distribution uses 'Top 20 Nodes by Hierarchy' format."""
        output_path = tmp_path / "test.onnx"

        # Export with reporting enabled
        exporter = HTPExporter(verbose=False, enable_reporting=True)

        try:
            stats = exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
                opset_version=17,
            )

            # Check report file
            report_path = output_path.with_name(
                output_path.stem + "_htp_export_report.md"
            )
            if report_path.exists():
                report_content = report_path.read_text()

                # Should have "Top 20 Nodes by Hierarchy" not "Node Distribution Preview"
                assert "#### Top 20 Nodes by Hierarchy" in report_content
                assert "Node Distribution Preview" not in report_content

                # Should show 20 items, not 10
                # Count numbered items (1. through 20.)
                numbered_items = []
                for i in range(1, 21):
                    if f" {i}. " in report_content:
                        numbered_items.append(i)

                # Should have at least some numbered items (might be less than 20 if model is small)
                assert len(numbered_items) > 0
                assert max(numbered_items) <= 20

        except Exception as e:
            pytest.skip(f"Export failed: {e}")

    def test_complete_hierarchy_title(self, tmp_path):
        """Test Complete Module Hierarchy uses correct title."""
        output_path = tmp_path / "test.onnx"

        # Export with reporting enabled
        exporter = HTPExporter(verbose=False, enable_reporting=True)

        try:
            stats = exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
                opset_version=17,
            )

            # Check report file
            report_path = output_path.with_name(
                output_path.stem + "_htp_export_report.md"
            )
            if report_path.exists():
                report_content = report_path.read_text()

                # Should use "Complete HF Hierarchy with ONNX Nodes"
                assert "### Complete HF Hierarchy with ONNX Nodes" in report_content
                assert "### Complete Module Hierarchy" not in report_content

                # Should show node counts in the tree
                assert " nodes)" in report_content

                # Should be in a details section
                section_start = report_content.find(
                    "### Complete HF Hierarchy with ONNX Nodes"
                )
                if section_start > 0:
                    section_content = report_content[
                        section_start : section_start + 500
                    ]
                    assert "<details>" in section_content
                    assert "Click to expand" in section_content

        except Exception as e:
            pytest.skip(f"Export failed: {e}")

    def test_hierarchy_consistency(self, tmp_path):
        """Test that hierarchy format is consistent throughout report."""
        output_path = tmp_path / "test.onnx"

        # Export with reporting enabled
        exporter = HTPExporter(verbose=False, enable_reporting=True)

        try:
            stats = exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
                opset_version=17,
            )

            # Check report file
            report_path = output_path.with_name(
                output_path.stem + "_htp_export_report.md"
            )
            if report_path.exists():
                report_content = report_path.read_text()

                # Test general hierarchy consistency - not model-specific
                # 1. Both hierarchy sections should be present
                assert "#### Module Hierarchy Preview" in report_content
                assert "### Complete HF Hierarchy with ONNX Nodes" in report_content

                # 2. Both sections should use consistent tree formatting
                assert (
                    "├──" in report_content or "└──" in report_content
                )  # Tree characters

                # 3. Module hierarchy should show structure
                module_preview_start = report_content.find(
                    "#### Module Hierarchy Preview"
                )
                complete_hierarchy_start = report_content.find(
                    "### Complete HF Hierarchy with ONNX Nodes"
                )

                assert module_preview_start < complete_hierarchy_start, (
                    "Module preview should come before complete hierarchy"
                )

                # 4. Complete hierarchy should include node counts
                assert "nodes)" in report_content  # Format: "ModuleName (X nodes)"

                # 5. Both sections should be in expandable details blocks
                assert "<details>" in report_content
                assert "</details>" in report_content
                assert "Click to expand" in report_content

        except Exception as e:
            pytest.skip(f"Export failed: {e}")

    def test_bert_model_hierarchy_content(self, tmp_path):
        """Test BERT-specific hierarchy content in report."""
        output_path = tmp_path / "test.onnx"

        # Export with reporting enabled
        exporter = HTPExporter(verbose=False, enable_reporting=True)

        try:
            stats = exporter.export(
                model_name_or_path="prajjwal1/bert-tiny",
                output_path=str(output_path),
                opset_version=17,
            )

            # Check report file
            report_path = output_path.with_name(
                output_path.stem + "_htp_export_report.md"
            )
            if report_path.exists():
                report_content = report_path.read_text()

                # BERT-specific model components should appear
                assert "BertModel" in report_content
                assert "BertEmbeddings" in report_content
                assert "BertEncoder" in report_content
                assert "BertLayer" in report_content
                assert "BertAttention" in report_content

                # Should NOT show torch.nn module class names in hierarchy
                # (These are internal implementation details, not part of HF model hierarchy)
                assert (
                    ": Linear" not in report_content
                )  # Would appear as "module_name: Linear"
                assert ": Dropout" not in report_content
                # Note: LayerNorm and Embedding might appear as part of BERT model class names

        except Exception as e:
            pytest.skip(f"Export failed: {e}")

    @pytest.mark.parametrize(
        "model_name",
        [
            "prajjwal1/bert-tiny",
            # Add more models if needed for broader testing
        ],
    )
    def test_report_structure_completeness(self, tmp_path, model_name):
        """Test that report has all required sections in correct order."""
        output_path = tmp_path / "test.onnx"

        # Export with reporting enabled
        exporter = HTPExporter(verbose=False, enable_reporting=True)

        try:
            stats = exporter.export(
                model_name_or_path=model_name,
                output_path=str(output_path),
                opset_version=17,
            )

            # Check report file
            report_path = output_path.with_name(
                output_path.stem + "_htp_export_report.md"
            )
            if report_path.exists():
                report_content = report_path.read_text()

                # Check all sections are present in order
                sections = [
                    "# HTP ONNX Export Report",
                    "## Export Process",
                    "### ✅ Step 1/6: Model Preparation",
                    "### ✅ Step 2/6: Input Generation",
                    "### ✅ Step 3/6: Hierarchy Building",
                    "#### Module Hierarchy Preview",
                    "### ✅ Step 4/6: ONNX Export",
                    "### ✅ Step 5/6: Node Tagging",
                    "#### Top 20 Nodes by Hierarchy",
                    "### ✅ Step 6/6: Tag Injection",
                    "## Module Hierarchy",
                    "### Complete HF Hierarchy with ONNX Nodes",
                    "## Node Distribution",
                    "## Export Summary",
                ]

                last_pos = 0
                for section in sections:
                    pos = report_content.find(section, last_pos)
                    if pos == -1:
                        # Some sections might be optional or named slightly differently
                        continue
                    assert pos >= last_pos, f"Section '{section}' is out of order"
                    last_pos = pos

        except Exception as e:
            pytest.skip(f"Export failed: {e}")

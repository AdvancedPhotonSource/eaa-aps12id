from pathlib import Path

import numpy as np

from eaa_core.gui.html import launch_html_webui_subprocess
from eaa_core.tool.mcp_client import MCPTool

from eaa_aps12id.task_managers.spatial_saxs_sampling import (
    SpatialSAXSAdaptiveSamplingTaskManager,
)
from eaa_aps12id.tools.aps12id_saxs import APS12IDSAXSAcquisitionTool


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_URL = "http://127.0.0.1:8010"


def main() -> None:
    mcp_tool = MCPTool(
        {
            "mcpServers": {
                "aps12id_saxs": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "http",
                }
            }
        }
    )
    acquisition_tool = APS12IDSAXSAcquisitionTool(mcp_tool)

    task_manager = SpatialSAXSAdaptiveSamplingTaskManager(
        acquisition_tool=acquisition_tool,
        checkpoint_db_path=str(PROJECT_ROOT / "checkpoint.sqlite"),
        transcript_db_path=str(PROJECT_ROOT / "transcript.sqlite"),
        use_webui=True,
        webui_runtime_host="127.0.0.1",
        webui_runtime_port=8010,
    )

    task_manager.start_webui_runtime()
    webui_process = launch_html_webui_subprocess(
        RUNTIME_URL,
        host="127.0.0.1",
        port=8008,
        title="Adaptive SAXS sampling",
    )
    print("WebUI: http://127.0.0.1:8008")
    try:
        x_grid, y_grid = np.meshgrid(
            np.linspace(-1.5, 1.5, 60),
            np.linspace(-1.9, -1.3, 12),
            indexing="xy",
        )
        task_manager.run(
            candidate_positions=np.column_stack((y_grid.ravel(), x_grid.ravel())),
            q_min=0.01,
            q_max=0.8,
            num_q_points=1000,
            num_initial_samples=10,
            max_measurements=50,
            num_initial_peaks=5,
            max_peaks_in_dict=10,
            w_peak=0.8,
            w_g=0.5,
            non_position_kwargs_for_acquisition_tool={
                "q_min": 3.75000000e-03,
                "q_max": 8.76000000e-01,
            },
        )
    finally:
        webui_process.terminate()
        task_manager.stop_webui_runtime()


if __name__ == "__main__":
    main()

"""Launch the spatial SAXS adaptive-sampling backend without starting a WebUI."""

from pathlib import Path

import numpy as np

from eaa_core.tool.mcp_client import MCPTool

from eaa_aps12id.task_managers.spatial_saxs_sampling import (
    SpatialSAXSAdaptiveSamplingTaskManager,
)
from eaa_aps12id.tools.aps12id_saxs import APS12IDSAXSAcquisitionTool


SCRIPT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Configure and run the spatial SAXS adaptive-sampling backend.
    
    The MCP server hosts the data acquisition tools. If the MCP server is not found
    at the port, consult the user.
    """
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
        checkpoint_db_path=str(SCRIPT_ROOT / "checkpoint.sqlite"),
        transcript_db_path=str(SCRIPT_ROOT / "transcript.sqlite"),
    )

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


if __name__ == "__main__":
    main()

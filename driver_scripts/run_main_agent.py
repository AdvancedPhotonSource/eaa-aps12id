import os
from pathlib import Path

from eaa_core.api.llm_config import ArgoConfig
from eaa_core.gui.html import launch_html_webui_subprocess
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.tool.mcp_client import MCPTool

from eaa_aps12id.task_managers.spatial_saxs_sampling import (
    SpatialSAXSAdaptiveSamplingTaskManager,
)
from eaa_aps12id.tools.aps12id_saxs import APS12IDSAXSAcquisitionTool


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_URL = "http://127.0.0.1:8010"


def main() -> None:
    llm_config = ArgoConfig(
        model="gpt55",
        api_key=os.environ["ARGO_API_KEY"],
    )
    aps12id_mcp_tool = MCPTool(
        {
            "mcpServers": {
                "aps12id": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "http",
                }
            }
        }
    )

    spatial_sampling_task_manager = SpatialSAXSAdaptiveSamplingTaskManager(
        name="spatial_saxs_adaptive_sampling",
        acquisition_tool=APS12IDSAXSAcquisitionTool(aps12id_mcp_tool),
        checkpoint_db_path=None,
        transcript_db_path=None,
    )
    main_task_manager = BaseTaskManager(
        llm_config=llm_config,
        tools=[aps12id_mcp_tool],
        checkpoint_db_path=str(PROJECT_ROOT / "checkpoint.sqlite"),
        transcript_db_path=str(PROJECT_ROOT / "transcript.sqlite"),
        use_webui=True,
        webui_runtime_host="127.0.0.1",
        webui_runtime_port=8010,
    )
    main_task_manager.tool_manager.subagent_tool.add_task_managers(
        spatial_sampling_task_manager
    )

    main_task_manager.start_webui_runtime()
    webui_process = launch_html_webui_subprocess(
        RUNTIME_URL,
        host="127.0.0.1",
        port=8008,
        title="APS 12-ID experiment assistant",
    )
    print("WebUI: http://127.0.0.1:8008")
    try:
        main_task_manager.run_conversation()
    finally:
        webui_process.terminate()
        main_task_manager.stop_webui_runtime()


if __name__ == "__main__":
    main()

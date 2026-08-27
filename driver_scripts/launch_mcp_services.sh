#!/usr/bin/env bash
# Launch the MCP servers used by the EAA agent and the services they depend on.
#
# Set these two variables to the repositories' locations on the target machine,
# either here or in the environment before running this script:
#
#   APS12_SAXSDAQ_DIR=/path/to/APS12_SAXSDaq \
#   AFL_STEERING_MCP_DIR=/path/to/AFL_steering_MCP \
#     ./driver_scripts/launch_mcp_services.sh
#
# Each process runs in a detached GNU Screen session. For example:
#
#   screen -ls
#   screen -r eaa-aps12-mcp
#   screen -S eaa-aps12-mcp -X quit

# Do not enable `set -e`: one unavailable service must not prevent the remaining
# independent services from being considered for launch.

APS12_SAXSDAQ_DIR="${APS12_SAXSDAQ_DIR:-/path/to/APS12_SAXSDaq}"
AFL_STEERING_MCP_DIR="${AFL_STEERING_MCP_DIR:-/path/to/AFL_steering_MCP}"

# These ports match APS12_SAXSDaq/json/autonomous_config.json and the agent
# driver scripts in this repository. Keep the GUI ports synchronized with that
# JSON file if the beamline configuration is changed.
MQTT_PORT=1883
APS12_MAIN_ZMQ_PORT=9876
APS12_MAIN_AGENT_ZMQ_PORT=9886
APS12_MH_ZMQ_PORT=9877
APS12_MH_AGENT_ZMQ_PORT=9887
APS12_MCP_PORT=8000
AFL_MCP_PORT=8050

SCREEN_PREFIX="${MCP_SCREEN_PREFIX:-eaa}"
APS12_BEAMLINE="${APS12_BEAMLINE:-12IDB}"
AFL_CONFIG="${AFL_CONFIG:-configs/manos.yaml}"
MOSQUITTO_BIN="${MOSQUITTO_BIN:-mosquitto}"
APS12_PYTHON="${APS12_PYTHON:-${APS12_SAXSDAQ_DIR}/.venv/bin/python}"

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

command_exists() {
    if [[ "$1" == */* ]]; then
        [[ -x "$1" ]]
    else
        command -v "$1" >/dev/null 2>&1
    fi
}

screen_session_exists() {
    screen -ls 2>/dev/null | grep -Fq ".${1}"
}

wait_for_port() {
    local service="$1"
    local port="$2"
    local attempt

    for attempt in {1..20}; do
        if port_is_listening "$port"; then
            return 0
        fi
        sleep 0.1
    done
    warn "${service} did not start listening on TCP port ${port} within 2 seconds"
    return 1
}

port_is_listening() {
    local port="$1"
    local hex_port
    local sockets=()

    printf -v hex_port '%04X' "$port"
    [[ -r /proc/net/tcp ]] && sockets+=(/proc/net/tcp)
    [[ -r /proc/net/tcp6 ]] && sockets+=(/proc/net/tcp6)
    if ((${#sockets[@]})); then
        awk -v port=":${hex_port}" \
            '$2 ~ port "$" && $4 == "0A" { found = 1 } END { exit !found }' \
            "${sockets[@]}"
        return
    fi

    if command -v ss >/dev/null 2>&1; then
        [[ -n "$(ss -H -ltn "sport = :${port}" 2>/dev/null)" ]]
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | awk -v port=":${port}" \
            '$4 ~ port "$" { found = 1 } END { exit !found }'
    else
        return 1
    fi
}

ports_are_available() {
    local service="$1"
    shift
    local port

    for port in "$@"; do
        if port_is_listening "$port"; then
            warn "not launching ${service}: TCP port ${port} is already in use"
            return 1
        fi
    done
    return 0
}

start_screen_session() {
    local session="$1"
    local working_dir="$2"
    shift 2

    if [[ ! -d "$working_dir" ]]; then
        warn "not launching ${session}: directory does not exist: ${working_dir}"
        return 1
    fi
    if screen_session_exists "$session"; then
        warn "not launching ${session}: screen session already exists"
        return 1
    fi

    if screen -dmS "$session" bash -lc \
        'cd "$1" || exit 1; shift; exec "$@"' bash "$working_dir" "$@"; then
        printf 'Started %-24s (screen session: %s)\n' "$session" "$session"
    else
        warn "failed to launch screen session ${session}"
        return 1
    fi
}

check_declared_ports() {
    local declarations=(
        "MQTT:${MQTT_PORT}"
        "APS12 main ZMQ:${APS12_MAIN_ZMQ_PORT}"
        "APS12 main agent ZMQ:${APS12_MAIN_AGENT_ZMQ_PORT}"
        "APS12 multi-heater ZMQ:${APS12_MH_ZMQ_PORT}"
        "APS12 multi-heater agent ZMQ:${APS12_MH_AGENT_ZMQ_PORT}"
        "APS12 MCP:${APS12_MCP_PORT}"
        "AFL MCP:${AFL_MCP_PORT}"
    )
    local seen=' '
    local declaration name port

    for declaration in "${declarations[@]}"; do
        name="${declaration%:*}"
        port="${declaration##*:}"
        if [[ "$seen" == *" ${port} "* ]]; then
            warn "port configuration conflict: ${name} also uses TCP port ${port}"
            return 1
        fi
        seen+="${port} "
    done
    return 0
}

if ! command_exists screen; then
    warn "GNU Screen is not installed; no services were launched"
    exit 0
fi

if ! check_declared_ports; then
    warn "no services were launched; assign unique ports first"
    exit 0
fi

# APS12 MQTT broker. APS12_SAXSDaq's start.sh also checks for this broker, but
# launching it explicitly keeps it independently visible and manageable.
if [[ ! -f "${APS12_SAXSDAQ_DIR}/tools/mosquitto.conf" ]]; then
    warn "not launching ${SCREEN_PREFIX}-aps12-mqtt: missing ${APS12_SAXSDAQ_DIR}/tools/mosquitto.conf"
elif ! command_exists "$MOSQUITTO_BIN"; then
    warn "not launching ${SCREEN_PREFIX}-aps12-mqtt: executable not found: ${MOSQUITTO_BIN}"
elif ports_are_available "${SCREEN_PREFIX}-aps12-mqtt" "$MQTT_PORT"; then
    if start_screen_session \
        "${SCREEN_PREFIX}-aps12-mqtt" "$APS12_SAXSDAQ_DIR" \
        "$MOSQUITTO_BIN" -c tools/mosquitto.conf; then
        wait_for_port "${SCREEN_PREFIX}-aps12-mqtt" "$MQTT_PORT"
    fi
fi

# The main DAQ and multi-heater processes provide distinct operator and agent
# ZMQ endpoints. APS12_SIMULATION and other APS12_* variables are inherited if
# the caller exports them before running this launcher.
if [[ ! -x "${APS12_SAXSDAQ_DIR}/start.sh" ]]; then
    warn "not launching APS12 GUI services: missing executable ${APS12_SAXSDAQ_DIR}/start.sh"
else
    if ports_are_available "${SCREEN_PREFIX}-aps12-main" \
        "$APS12_MAIN_ZMQ_PORT" "$APS12_MAIN_AGENT_ZMQ_PORT"; then
        start_screen_session \
            "${SCREEN_PREFIX}-aps12-main" "$APS12_SAXSDAQ_DIR" \
            env APS12_BEAMLINE="$APS12_BEAMLINE" ./start.sh main
    fi

    if ports_are_available "${SCREEN_PREFIX}-aps12-mh" \
        "$APS12_MH_ZMQ_PORT" "$APS12_MH_AGENT_ZMQ_PORT"; then
        start_screen_session \
            "${SCREEN_PREFIX}-aps12-mh" "$APS12_SAXSDAQ_DIR" \
            env APS12_BEAMLINE="$APS12_BEAMLINE" ./start.sh mh
    fi
fi

# APS12 HTTP MCP adapter. FASTMCP_* controls its HTTP listener while APS12_*
# selects the ZMQ and MQTT endpoints used by the adapter.
if [[ ! -f "${APS12_SAXSDAQ_DIR}/mcp_server/id12_mcp.py" ]]; then
    warn "not launching ${SCREEN_PREFIX}-aps12-mcp: MCP server source not found"
elif ! command_exists "$APS12_PYTHON"; then
    warn "not launching ${SCREEN_PREFIX}-aps12-mcp: Python executable not found: ${APS12_PYTHON}"
elif ports_are_available "${SCREEN_PREFIX}-aps12-mcp" "$APS12_MCP_PORT"; then
    start_screen_session \
        "${SCREEN_PREFIX}-aps12-mcp" "$(dirname "$APS12_SAXSDAQ_DIR")" \
        env \
        APS12_BEAMLINE="$APS12_BEAMLINE" \
        APS12_AGENT_PORT="$APS12_MAIN_AGENT_ZMQ_PORT" \
        APS12_MH_AGENT_PORT="$APS12_MH_AGENT_ZMQ_PORT" \
        APS12_MQTT_PORT="$MQTT_PORT" \
        FASTMCP_HOST=127.0.0.1 \
        FASTMCP_PORT="$APS12_MCP_PORT" \
        FASTMCP_STREAMABLE_HTTP_PATH=/mcp \
        "$APS12_PYTHON" -c \
        'from APS12_SAXSDaq.mcp_server.id12_mcp import mcp; mcp.run(transport="streamable-http")'
fi

# AFL steering HTTP MCP server. Its command-line port override guarantees that
# it does not inherit a conflicting value from the selected scientific config.
if [[ ! -d "$AFL_STEERING_MCP_DIR" ]]; then
    warn "not launching ${SCREEN_PREFIX}-afl-mcp: directory does not exist: ${AFL_STEERING_MCP_DIR}"
elif [[ ! -f "${AFL_STEERING_MCP_DIR}/${AFL_CONFIG}" ]]; then
    warn "not launching ${SCREEN_PREFIX}-afl-mcp: config does not exist: ${AFL_STEERING_MCP_DIR}/${AFL_CONFIG}"
elif ! command_exists uv; then
    warn "not launching ${SCREEN_PREFIX}-afl-mcp: uv is not installed"
elif ports_are_available "${SCREEN_PREFIX}-afl-mcp" "$AFL_MCP_PORT"; then
    start_screen_session \
        "${SCREEN_PREFIX}-afl-mcp" "$AFL_STEERING_MCP_DIR" \
        uv run afl-steering-mcp \
        --config "$AFL_CONFIG" --host 127.0.0.1 --port "$AFL_MCP_PORT" --path /mcp
fi

printf '\nManage services with: screen -ls | screen -r <name> | screen -S <name> -X quit\n'

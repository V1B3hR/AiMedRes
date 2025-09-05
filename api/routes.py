@app.route('/api/v1/labyrinth/reason', methods=['POST'])
def labyrinth_reason():
    data = request.get_json()
    agent_idx = int(data.get("agent_idx", 0))
    task = data.get("task")
    agent = current_app.labyrinth_agents[agent_idx]
    result = agent.reason(task)
    return jsonify(result)

@app.route('/api/v1/labyrinth/state', methods=['GET'])
def labyrinth_state():
    states = [agent.get_state() for agent in current_app.labyrinth_agents]
    return jsonify({"labyrinth_agents": states})

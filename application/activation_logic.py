import time


def build_current_task(stable_step_id, step_progress, trajectory_config):
    step_id = int(stable_step_id)
    trajectory_cfg = trajectory_config.get(step_id)
    robot_capable = (
        trajectory_cfg is not None
        and trajectory_cfg.get("robot_capable", False)
    )

    return {
        "step_id": step_id,
        "step_progress": float(step_progress),
        "robot_capable": robot_capable,
        "suggested_action": (
            trajectory_cfg.get("suggested_action", "prepare_assistance")
            if trajectory_cfg is not None
            else "wait"
        ),
    }


def permission_should_be_requested(current_task, progress_threshold):
    return (
        current_task["robot_capable"]
        and current_task["step_progress"] >= progress_threshold
    )


def command_grants_permission(command):
    return command == "y"


def add_pending_task(pending_task_pool, current_task, command, now_fn=time.time):
    status = "timeout_pending" if command is None or command == "" else "rejected_pending"

    pending_task_pool.append({
        "step_id": current_task["step_id"],
        "step_progress": current_task["step_progress"],
        "suggested_action": current_task["suggested_action"],
        "created_at": now_fn(),
        "status": status,
    })

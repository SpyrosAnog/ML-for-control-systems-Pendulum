# Reward function used to train a2c_ref_track_v2.zip
# Saved at 2026-06-14T20:09:50
# Weights: {'swing_up': 0.5, 'capture': 8.0, 'top_speed_penalty': 1.0, 'omega_penalty': 0.02, 'action_penalty': 0.005, 'stall_penalty': 0.5}

def external_ref_reward(
    theta: float, omega: float, action: float, umax: float, theta_ref: float
) -> float:
    """Reward for reference tracking.

    theta_ref is the target angle for this episode (π ± up to 15°).
    The swing-up term still uses raw height so the policy climbs from the bottom.
    The capture / speed terms are centred on theta_ref so the policy settles there.
    """
    theta_wrapped = wrap_angle(theta)

    # General height reward — same as swing-up policy, reference-agnostic
    # Keeps pushing the pendulum toward the top region during swing-up
    swing_up_reward = ((1.0 - np.cos(theta_wrapped)) / 2.0) ** 2

    # Deviation from the reference (0 when exactly at theta_ref)
    ref_error = abs(wrap_angle(theta - theta_ref))

    # Capture bonus: rewards being near theta_ref AND slow
    top_gate      = np.exp(-(ref_error**2) / (2 * REWARD_PARAMS["top_gate_sigma"]**2))
    capture_bonus = (np.exp(-(ref_error**2) / (2 * REWARD_PARAMS["balance_sigma_angle"]**2))
                     * np.exp(-(omega**2) / (2 * REWARD_PARAMS["balance_sigma_omega"]**2)))

    top_speed_penalty = top_gate * (omega / REWARD_PARAMS["top_speed_sigma"]) ** 2
    omega_penalty     = (omega / REWARD_PARAMS["omega_penalty_denom"]) ** 2
    action_penalty    = (action / umax) ** 2

    # Stall penalty uses distance from π (not theta_ref) to force swing-up
    upright_error = np.pi - abs(theta_wrapped)
    stall_penalty = (np.exp(-(omega**2) / (2 * REWARD_PARAMS["stall_omega_sigma"]**2))
                     * (upright_error / np.pi) ** 2)

    return (
        REWARD_WEIGHTS["swing_up"]             * swing_up_reward
        + REWARD_WEIGHTS["capture"]            * capture_bonus
        - REWARD_WEIGHTS["top_speed_penalty"]  * top_speed_penalty
        - REWARD_WEIGHTS["omega_penalty"]      * omega_penalty
        - REWARD_WEIGHTS["action_penalty"]     * action_penalty
        - REWARD_WEIGHTS["stall_penalty"]      * stall_penalty
    )

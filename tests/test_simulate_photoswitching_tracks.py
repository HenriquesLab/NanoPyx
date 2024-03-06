from nanopyx.core.generate.simulate_photoswitching_time_tracks import simple_state_transition_model


def test_simple_state_transition_model(): #(plt):
    """Test the simple state transition model."""

    # Define the parameters
    n_particles = 100
    n_steps = 1000
    p_on = 0.01
    p_transient_off = 0.1
    p_permanent_off = 0.01

    # Simulate the particle field
    particle_field = simple_state_transition_model(
        n_particles, n_steps, p_on, p_transient_off, p_permanent_off, 1)

    # # Plot the results
    # plt.imshow(particle_field, interpolation='none')
    # plt.title('Particle Intensity Tracks')
    # plt.xlabel('Time-Traces')
    # plt.ylabel('Particle')


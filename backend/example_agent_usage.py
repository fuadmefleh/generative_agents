"""
Example: How to create agents with emergent personalities and life stories

This demonstrates the new agent creation system that generates:
- Life stories with 2-3 shaping events per age period
- Personality traits shaped by life events
- Detailed daily routines including mundane activities
- Granular activity tracking (using toilet, sleeping on bed, looking in fridge, etc.)
"""

from app.factory.agent_factory import AgentFactory
from app.models.agent import JobType
from app.models.models import Point2D


def create_agent_example():
    """Example: Create a single agent with a generated life story."""
    
    # Initialize the factory
    factory = AgentFactory()
    
    # Create an agent with emergent personality
    agent = factory.create_agent_with_life_story(
        agent_id="agent_001",
        name="Emma Rodriguez",
        location=Point2D(x=25.0, y=30.0),
        home_location_name="Emma's Apartment",
        age=32,
        occupation=JobType.ARTIST,
        cultural_background="Hispanic American",
        generate_life_story=True  # Set False to skip LLM generation for testing
    )
    
    # Access the agent's life story
    if agent.personality.life_story:
        print(f"📖 {agent.state.name}'s Life Story:")
        print(f"   Born in: {agent.personality.life_story.birthplace}")
        print(f"   Life events: {len(agent.personality.life_story.events)}")
        print(f"   Core values: {', '.join(agent.personality.life_story.core_values)}")
        print(f"   Aspirations: {', '.join(agent.personality.life_story.aspirations)}")
    
    # Check personality traits (shaped by life story)
    print(f"\n🧠 Personality (shaped by life events):")
    print(f"   {agent.personality.get_description()}")
    
    # View daily schedule
    print(f"\n📅 Daily Schedule ({len(agent.daily_schedule)} activities):")
    for activity in agent.daily_schedule[:5]:  # Show first 5
        time_str = f"{activity.start_hour:02d}:{activity.start_minute:02d}"
        print(f"   {time_str} - {activity.description}")
        if activity.specific_objects:
            print(f"           Using: {', '.join(activity.specific_objects)}")
    
    return agent


def create_multiple_agents_example():
    """Example: Create multiple diverse agents."""
    
    factory = AgentFactory()
    
    # Helper function for random locations
    import random
    def random_location():
        return Point2D(x=float(random.randint(10, 90)), y=float(random.randint(10, 90)))
    
    # Create 5 agents with diverse backgrounds
    agents = factory.create_multiple_agents(
        count=5,
        location_generator=random_location,
        home_locations=[
            "Hobbs Cafe",
            "The Rose and Crown Pub", 
            "Johnson Park",
            "City Library",
            "Main Street Apartments"
        ],
        occupations=[
            JobType.BARISTA,
            JobType.ARTIST,
            JobType.WRITER,
            JobType.TEACHER,
            JobType.MUSICIAN
        ],
        cultural_backgrounds=[
            "Urban American",
            "Rural Southern",
            "Coastal Pacific",
            "Midwest",
            "International"
        ]
    )
    
    print(f"\n✅ Created {len(agents)} agents with unique backgrounds")
    for agent in agents:
        print(f"   - {agent.state.name}: {agent.job.job_type.value if agent.job else 'unemployed'}")
    
    return agents


def track_agent_activities_example(agent):
    """Example: Track what an agent is doing throughout the day."""
    from datetime import datetime
    
    print(f"\n🔍 Tracking {agent.state.name}'s activities:")
    
    # Simulate different times of day
    times_to_check = [
        (7, 0, "Morning"),
        (12, 30, "Lunch"),
        (18, 0, "Evening"),
        (22, 0, "Night")
    ]
    
    for hour, minute, label in times_to_check:
        test_time = datetime(2024, 1, 15, hour, minute)
        day_of_week = test_time.weekday()
        
        # Check scheduled activity
        for activity in agent.daily_schedule:
            if activity.is_active_at(hour, minute, day_of_week):
                # Simulate starting the activity
                agent.state.start_scheduled_activity(activity, test_time)
                
                # Get detailed status - this tells you exactly what they're doing
                status = agent.state.get_detailed_status()
                
                print(f"\n   {label} ({hour:02d}:{minute:02d}):")
                print(f"   {status}")
                print(f"   Activity: {activity.activity_type.value}")
                
                # You can see specific objects being used
                if agent.state.objects_in_use:
                    print(f"   Objects: {', '.join(agent.state.objects_in_use)}")
                
                # You can see exact location
                if agent.state.current_sub_location:
                    print(f"   Location: {agent.state.current_location_name} > {agent.state.current_sub_location}")
                
                break


def integrate_with_simulation_example():
    """Example: How to integrate with your simulation loop."""
    
    # Create agents using the factory
    factory = AgentFactory()
    
    def get_spawn_location():
        return Point2D(x=50.0, y=50.0)
    
    agent = factory.create_agent_with_life_story(
        agent_id="sim_agent_001",
        name="Alex Chen",
        location=get_spawn_location(),
        home_location_name="Alex's Home",
        age=28,
        occupation=JobType.WRITER
    )
    
    # In your simulation tick/update loop:
    from datetime import datetime
    
    simulation_time = datetime(2024, 1, 15, 8, 30)  # 8:30 AM
    
    # Check and execute scheduled activities
    # This returns True if agent is following their schedule
    if agent.check_and_execute_scheduled_activity(simulation_time):
        # Agent is doing their scheduled activity
        print(f"\n🎬 In simulation at {simulation_time.strftime('%I:%M %p')}:")
        print(f"   {agent.state.get_detailed_status()}")
        
        # You know exactly what they're doing:
        print(f"\n   Details:")
        print(f"   - Action: {agent.state.current_activity_type.value if agent.state.current_activity_type else 'none'}")
        print(f"   - Where: {agent.state.current_location_name}")
        print(f"   - Sub-location: {agent.state.current_sub_location}")
        print(f"   - Using: {', '.join(agent.state.objects_in_use) if agent.state.objects_in_use else 'nothing'}")
        print(f"   - Status: {agent.state.status}")
        print(f"   - Sleeping: {agent.state.is_sleeping}")
        print(f"   - Busy: {agent.state.is_busy}")


if __name__ == "__main__":
    print("=" * 80)
    print("AGENT CREATION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Create single agent
    print("\n\n1️⃣  Creating a single agent with life story:")
    print("-" * 80)
    agent = create_agent_example()
    
    # Example 2: Create multiple agents
    print("\n\n2️⃣  Creating multiple diverse agents:")
    print("-" * 80)
    agents = create_multiple_agents_example()
    
    # Example 3: Track activities
    print("\n\n3️⃣  Tracking agent activities:")
    print("-" * 80)
    track_agent_activities_example(agent)
    
    # Example 4: Simulation integration
    print("\n\n4️⃣  Integration with simulation:")
    print("-" * 80)
    integrate_with_simulation_example()
    
    print("\n\n" + "=" * 80)
    print("✅ Examples completed!")
    print("=" * 80)

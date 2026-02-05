"""Test script to generate agents with life stories and verify routines."""
import sys
from pathlib import Path
from datetime import datetime
import json

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.factory.agent_factory import AgentFactory
from app.models.agent import JobType
from app.models.models import Point2D
from loguru import logger

# Configure logger
logger.add("agent_generation_test.log", rotation="10 MB")


def test_single_agent_generation():
    """Test generating a single agent with life story."""
    logger.info("=" * 80)
    logger.info("Testing single agent generation with life story")
    logger.info("=" * 80)
    
    factory = AgentFactory()
    
    # Create a test agent
    agent = factory.create_agent_with_life_story(
        agent_id="test_001",
        name="Alice Johnson",
        location=Point2D(x=10.0, y=10.0),
        home_location_name="Hobbs Cafe",
        age=28,
        occupation=JobType.BARISTA,
        cultural_background="Urban American",
        generate_life_story=True  # Set to False for quick testing
    )
    
    # Print agent details
    print("\n" + "=" * 80)
    print(f"AGENT: {agent.state.name}")
    print("=" * 80)
    
    # Personality
    print("\n📊 PERSONALITY TRAITS:")
    print(f"  Openness:          {agent.personality.openness:.2f}")
    print(f"  Conscientiousness: {agent.personality.conscientiousness:.2f}")
    print(f"  Extraversion:      {agent.personality.extraversion:.2f}")
    print(f"  Agreeableness:     {agent.personality.agreeableness:.2f}")
    print(f"  Neuroticism:       {agent.personality.neuroticism:.2f}")
    print(f"  Work Ethic:        {agent.personality.work_ethic:.2f}")
    print(f"  Spontaneity:       {agent.personality.spontaneity:.2f}")
    print(f"\n  Wake Hour:         {agent.personality.preferred_wake_hour}:00")
    print(f"  Sleep Hour:        {agent.personality.preferred_sleep_hour}:00")
    print(f"  Favorite Activities: {', '.join(agent.personality.favorite_activities)}")
    
    # Life story
    if agent.personality.life_story:
        life_story = agent.personality.life_story
        print(f"\n📖 LIFE STORY:")
        print(f"  Age: {life_story.current_age}")
        print(f"  Birthplace: {life_story.birthplace}")
        print(f"  Core Values: {', '.join(life_story.core_values)}")
        print(f"  Fears: {', '.join(life_story.fears)}")
        print(f"  Aspirations: {', '.join(life_story.aspirations)}")
        
        print(f"\n  Life Events ({len(life_story.events)} total):")
        for i, event in enumerate(life_story.events[:10], 1):  # Show first 10
            print(f"\n  {i}. Age {event.age} - {event.category.value.upper()}")
            print(f"     {event.description}")
            print(f"     Emotional Impact: {event.emotional_impact:+.2f}")
            if event.personality_impact.has_impact():
                impacts = []
                if abs(event.personality_impact.openness_delta) > 0.01:
                    impacts.append(f"openness {event.personality_impact.openness_delta:+.2f}")
                if abs(event.personality_impact.extraversion_delta) > 0.01:
                    impacts.append(f"extraversion {event.personality_impact.extraversion_delta:+.2f}")
                if abs(event.personality_impact.conscientiousness_delta) > 0.01:
                    impacts.append(f"conscientiousness {event.personality_impact.conscientiousness_delta:+.2f}")
                if impacts:
                    print(f"     Impact: {', '.join(impacts)}")
        
        if len(life_story.events) > 10:
            print(f"\n  ... and {len(life_story.events) - 10} more events")
        
        print(f"\n  Narrative Summary:")
        print(f"  {life_story.get_narrative_summary()}")
    
    # Job
    if agent.job:
        print(f"\n💼 JOB:")
        print(f"  Type: {agent.job.job_type.value}")
        print(f"  Workplace: {agent.job.workplace_location}")
        print(f"  Sub-location: {agent.job.workplace_sub_location}")
        print(f"  Satisfaction: {agent.job.job_satisfaction:.2f}")
    
    # Daily schedule
    print(f"\n📅 DAILY SCHEDULE ({len(agent.daily_schedule)} activities):")
    for activity in agent.daily_schedule:
        time_str = f"{activity.start_hour:02d}:{activity.start_minute:02d} - {activity.end_hour:02d}:{activity.end_minute:02d}"
        print(f"\n  {time_str} | {activity.activity_type.value.upper()}")
        print(f"    📍 {activity.location}", end="")
        if activity.sub_location:
            print(f" > {activity.sub_location}", end="")
        print()
        print(f"    📝 {activity.description}")
        if activity.specific_objects:
            print(f"    🔧 Using: {', '.join(activity.specific_objects)}")
        print(f"    ⚠️  Priority: {activity.priority}/10")
    
    print("\n" + "=" * 80)
    
    return agent


def test_multiple_agents():
    """Test generating multiple agents."""
    logger.info("=" * 80)
    logger.info("Testing multiple agent generation")
    logger.info("=" * 80)
    
    factory = AgentFactory()
    
    # Generate 3 agents with different backgrounds
    def random_location():
        import random
        return Point2D(x=float(random.randint(5, 50)), y=float(random.randint(5, 50)))
    
    agents = factory.create_multiple_agents(
        count=3,
        location_generator=random_location,
        home_locations=["Hobbs Cafe", "The Rose and Crown Pub", "Harvey Oak Supply Store"],
        occupations=[JobType.BARISTA, JobType.ARTIST, JobType.WRITER],
        cultural_backgrounds=["Urban American", "Rural Southern", "Coastal Pacific"]
    )
    
    print("\n" + "=" * 80)
    print("GENERATED AGENTS SUMMARY")
    print("=" * 80)
    
    for i, agent in enumerate(agents, 1):
        print(f"\n{i}. {agent.state.name}")
        print(f"   Age: {agent.personality.life_story.current_age if agent.personality.life_story else 'N/A'}")
        if agent.job:
            print(f"   Job: {agent.job.job_type.value}")
        print(f"   Personality: {agent.personality.get_description()}")
        if agent.personality.life_story:
            print(f"   Life Events: {len(agent.personality.life_story.events)}")
            print(f"   Values: {', '.join(agent.personality.life_story.core_values[:3])}")
        print(f"   Daily Activities: {len(agent.daily_schedule)}")
    
    print("\n" + "=" * 80)
    
    return agents


def test_agent_detailed_status():
    """Test the detailed status reporting."""
    logger.info("=" * 80)
    logger.info("Testing detailed status reporting")
    logger.info("=" * 80)
    
    factory = AgentFactory()
    
    agent = factory.create_agent_with_life_story(
        agent_id="test_status",
        name="Bob Smith",
        location=Point2D(x=15.0, y=15.0),
        home_location_name="home",
        age=35,
        occupation=JobType.CHEF,
        generate_life_story=False  # Quick test
    )
    
    print("\n" + "=" * 80)
    print("TESTING ACTIVITY TRACKING")
    print("=" * 80)
    
    # Simulate different times and check scheduled activities
    test_times = [
        datetime(2024, 1, 15, 7, 0),   # Morning wake up
        datetime(2024, 1, 15, 7, 30),  # Morning routine
        datetime(2024, 1, 15, 8, 0),   # Breakfast
        datetime(2024, 1, 15, 12, 0),  # Lunch
        datetime(2024, 1, 15, 18, 0),  # Dinner prep
        datetime(2024, 1, 15, 22, 0),  # Bedtime
    ]
    
    for test_time in test_times:
        # Check if there's a scheduled activity at this time
        hour = test_time.hour
        minute = test_time.minute
        day_of_week = test_time.weekday()
        
        matching_activities = [
            act for act in agent.daily_schedule
            if act.is_active_at(hour, minute, day_of_week)
        ]
        
        print(f"\n⏰ {test_time.strftime('%I:%M %p')}")
        if matching_activities:
            activity = matching_activities[0]
            # Simulate starting the activity
            agent.state.start_scheduled_activity(activity, test_time)
            
            # Get detailed status
            status = agent.state.get_detailed_status()
            print(f"   {status}")
            print(f"   Activity Type: {activity.activity_type.value}")
            if activity.specific_objects:
                print(f"   Objects: {', '.join(activity.specific_objects)}")
        else:
            print("   No scheduled activity")
    
    print("\n" + "=" * 80)


def save_agent_to_json(agent, filename="test_agent.json"):
    """Save agent data to JSON for inspection."""
    data = {
        "name": agent.state.name,
        "age": agent.personality.life_story.current_age if agent.personality.life_story else None,
        "personality": {
            "openness": agent.personality.openness,
            "conscientiousness": agent.personality.conscientiousness,
            "extraversion": agent.personality.extraversion,
            "agreeableness": agent.personality.agreeableness,
            "neuroticism": agent.personality.neuroticism,
        },
        "life_story": None,
        "daily_schedule": [
            {
                "time": f"{act.start_hour:02d}:{act.start_minute:02d}",
                "activity": act.activity_type.value,
                "description": act.description,
                "location": act.location,
                "objects": act.specific_objects
            }
            for act in agent.daily_schedule
        ]
    }
    
    if agent.personality.life_story:
        data["life_story"] = {
            "birthplace": agent.personality.life_story.birthplace,
            "events": [
                {
                    "age": e.age,
                    "category": e.category.value,
                    "description": e.description,
                    "emotional_impact": e.emotional_impact
                }
                for e in agent.personality.life_story.events
            ],
            "core_values": agent.personality.life_story.core_values,
            "fears": agent.personality.life_story.fears,
            "aspirations": agent.personality.life_story.aspirations
        }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Agent data saved to {filename}")


if __name__ == "__main__":
    print("\n🧪 AGENT GENERATION TEST SUITE")
    print("=" * 80)
    
    # Test 1: Single agent with full life story
    print("\n\n🔹 TEST 1: Single Agent Generation")
    agent = test_single_agent_generation()
    save_agent_to_json(agent, "test_agent_full.json")
    
    # Test 2: Multiple agents
    print("\n\n🔹 TEST 2: Multiple Agents Generation")
    agents = test_multiple_agents()
    
    # Test 3: Activity tracking
    print("\n\n🔹 TEST 3: Activity Tracking")
    test_agent_detailed_status()
    
    print("\n\n✅ ALL TESTS COMPLETED")
    print("=" * 80)

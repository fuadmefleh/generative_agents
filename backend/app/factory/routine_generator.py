"""Generator for creating detailed daily routines for agents."""
import random
from typing import List, Optional
from datetime import time
from app.models.agent import (
    ScheduledActivity, ActionType, AgentPersonality, JobType, AgentJob
)
from app.models.life_story import LifeStory
from loguru import logger


class RoutineGenerator:
    """Generates detailed daily routines for agents based on personality and life story."""
    
    @staticmethod
    def generate_daily_routine(
        personality: AgentPersonality,
        job: Optional[AgentJob] = None,
        life_story: Optional[LifeStory] = None,
        home_location: str = "home"
    ) -> List[ScheduledActivity]:
        """
        Generate a complete daily routine with detailed activities.
        
        Args:
            personality: Agent's personality profile
            job: Optional job information
            life_story: Optional life story for context
            home_location: Agent's home location name
            
        Returns:
            List of scheduled activities throughout the day
        """
        routine = []
        
        # Morning routine
        morning_activities = RoutineGenerator._generate_morning_routine(
            personality, home_location
        )
        routine.extend(morning_activities)
        
        # Work/day activities
        if job:
            work_activities = RoutineGenerator._generate_work_routine(
                personality, job
            )
            routine.extend(work_activities)
        else:
            # If unemployed, generate daytime activities
            day_activities = RoutineGenerator._generate_daytime_routine(
                personality, home_location, life_story
            )
            routine.extend(day_activities)
        
        # Evening routine
        evening_activities = RoutineGenerator._generate_evening_routine(
            personality, home_location, life_story
        )
        routine.extend(evening_activities)
        
        # Night routine
        night_activities = RoutineGenerator._generate_night_routine(
            personality, home_location
        )
        routine.extend(night_activities)
        
        logger.info(f"Generated {len(routine)} scheduled activities for agent")
        return routine
    
    @staticmethod
    def _generate_morning_routine(
        personality: AgentPersonality,
        home_location: str
    ) -> List[ScheduledActivity]:
        """Generate morning routine from waking up to leaving home."""
        activities = []
        
        wake_hour = personality.preferred_wake_hour
        wake_minute = random.randint(0, 30)
        
        # Wake up
        activities.append(ScheduledActivity(
            start_hour=wake_hour,
            start_minute=wake_minute,
            end_hour=wake_hour,
            end_minute=min(wake_minute + 5, 59),
            activity_type=ActionType.WAKE_UP,
            location=home_location,
            sub_location="bedroom",
            description="Waking up and getting out of bed",
            specific_objects=["bed"],
            priority=9
        ))
        
        # Use toilet
        current_hour = wake_hour
        current_minute = wake_minute + 5
        if current_minute >= 60:
            current_hour += 1
            current_minute -= 60
        
        activities.append(ScheduledActivity(
            start_hour=current_hour,
            start_minute=current_minute,
            end_hour=current_hour,
            end_minute=min(current_minute + 5, 59),
            activity_type=ActionType.USE_TOILET,
            location=home_location,
            sub_location="bathroom",
            description="Using the bathroom",
            specific_objects=["toilet"],
            priority=9
        ))
        
        # Shower routine
        current_minute += 5
        if current_minute >= 60:
            current_hour += 1
            current_minute -= 60
        
        shower_duration = 15 if personality.conscientiousness > 0.6 else 10
        activities.append(ScheduledActivity(
            start_hour=current_hour,
            start_minute=current_minute,
            end_hour=current_hour,
            end_minute=min(current_minute + shower_duration, 59),
            activity_type=ActionType.SHOWER,
            location=home_location,
            sub_location="bathroom",
            description="Taking a shower",
            specific_objects=["shower"],
            sub_activities=[ActionType.BRUSH_TEETH],
            priority=8
        ))
        
        # Get dressed
        current_minute += shower_duration
        if current_minute >= 60:
            current_hour += 1
            current_minute -= 60
        
        activities.append(ScheduledActivity(
            start_hour=current_hour,
            start_minute=current_minute,
            end_hour=current_hour,
            end_minute=min(current_minute + 10, 59),
            activity_type=ActionType.GET_DRESSED,
            location=home_location,
            sub_location="bedroom",
            description="Getting dressed for the day",
            specific_objects=["closet", "mirror"],
            priority=8
        ))
        
        # Breakfast routine
        current_minute += 10
        if current_minute >= 60:
            current_hour += 1
            current_minute -= 60
        
        # Look in fridge
        activities.append(ScheduledActivity(
            start_hour=current_hour,
            start_minute=current_minute,
            end_hour=current_hour,
            end_minute=min(current_minute + 2, 59),
            activity_type=ActionType.LOOK_IN_FRIDGE,
            location=home_location,
            sub_location="kitchen",
            description="Checking the fridge for breakfast items",
            specific_objects=["fridge"],
            priority=7
        ))
        
        current_minute += 2
        if current_minute >= 60:
            current_hour += 1
            current_minute -= 60
        
        # Make coffee/breakfast
        cook_duration = 20 if personality.conscientiousness > 0.5 else 10
        activities.append(ScheduledActivity(
            start_hour=current_hour,
            start_minute=current_minute,
            end_hour=current_hour,
            end_minute=min(current_minute + cook_duration, 59),
            activity_type=ActionType.MAKE_COFFEE,
            location=home_location,
            sub_location="kitchen",
            description="Making coffee and preparing breakfast",
            specific_objects=["coffee_maker", "stove", "cupboard"],
            sub_activities=[ActionType.COOK, ActionType.LOOK_IN_CUPBOARD],
            priority=7
        ))
        
        current_minute += cook_duration
        if current_minute >= 60:
            current_hour += 1
            current_minute -= 60
        
        # Eat breakfast
        activities.append(ScheduledActivity(
            start_hour=current_hour,
            start_minute=current_minute,
            end_hour=current_hour,
            end_minute=min(current_minute + 20, 59),
            activity_type=ActionType.EAT,
            location=home_location,
            sub_location="kitchen",
            description="Eating breakfast",
            specific_objects=["table", "chair"],
            sub_activities=[ActionType.DRINK],
            priority=8
        ))
        
        return activities
    
    @staticmethod
    def _generate_work_routine(
        personality: AgentPersonality,
        job: AgentJob
    ) -> List[ScheduledActivity]:
        """Generate work-related activities."""
        activities = []
        
        # Use the job's work schedule if available
        if job.work_schedule:
            return job.work_schedule
        
        # Otherwise, generate default work schedule based on job type
        if job.job_type == JobType.BARISTA:
            work_start = 7
            work_end = 15
        elif job.job_type == JobType.TEACHER:
            work_start = 8
            work_end = 16
        elif job.job_type == JobType.STUDENT:
            work_start = 9
            work_end = 15
        elif job.job_type == JobType.ARTIST:
            work_start = 10
            work_end = 18
        else:
            work_start = 9
            work_end = 17
        
        activities.append(ScheduledActivity(
            start_hour=work_start,
            start_minute=0,
            end_hour=work_end,
            end_minute=0,
            activity_type=ActionType.WORK,
            location=job.workplace_location,
            sub_location=job.workplace_sub_location,
            description=f"Working as {job.job_type.value}",
            priority=9,
            days_of_week=[0, 1, 2, 3, 4]  # Monday-Friday
        ))
        
        # Lunch break
        lunch_hour = work_start + (work_end - work_start) // 2
        activities.append(ScheduledActivity(
            start_hour=lunch_hour,
            start_minute=0,
            end_hour=lunch_hour,
            end_minute=30,
            activity_type=ActionType.EAT,
            location=job.workplace_location,
            sub_location="break room",
            description="Lunch break",
            specific_objects=["table", "fridge", "microwave"],
            priority=8,
            days_of_week=[0, 1, 2, 3, 4]
        ))
        
        return activities
    
    @staticmethod
    def _generate_daytime_routine(
        personality: AgentPersonality,
        home_location: str,
        life_story: Optional[LifeStory]
    ) -> List[ScheduledActivity]:
        """Generate daytime activities for agents without jobs."""
        activities = []
        
        # Morning activities (10 AM - 12 PM)
        if "reading" in personality.favorite_activities:
            activities.append(ScheduledActivity(
                start_hour=10,
                start_minute=0,
                end_hour=11,
                end_minute=30,
                activity_type=ActionType.READ_BOOK,
                location=home_location,
                sub_location="living room",
                description="Reading a book",
                specific_objects=["couch", "bookshelf"],
                priority=5
            ))
        
        # Midday (12 PM - 2 PM) - lunch
        activities.append(ScheduledActivity(
            start_hour=12,
            start_minute=0,
            end_hour=12,
            end_minute=5,
            activity_type=ActionType.LOOK_IN_FRIDGE,
            location=home_location,
            sub_location="kitchen",
            description="Looking in fridge for lunch",
            specific_objects=["fridge"],
            priority=7
        ))
        
        activities.append(ScheduledActivity(
            start_hour=12,
            start_minute=5,
            end_hour=12,
            end_minute=35,
            activity_type=ActionType.COOK,
            location=home_location,
            sub_location="kitchen",
            description="Preparing lunch",
            specific_objects=["stove", "counter", "cupboard"],
            priority=7
        ))
        
        activities.append(ScheduledActivity(
            start_hour=12,
            start_minute=35,
            end_hour=13,
            end_minute=15,
            activity_type=ActionType.EAT,
            location=home_location,
            sub_location="kitchen",
            description="Eating lunch",
            specific_objects=["table"],
            priority=8
        ))
        
        # Afternoon activities
        if personality.extraversion > 0.6:
            # Social agent - go out
            activities.append(ScheduledActivity(
                start_hour=14,
                start_minute=0,
                end_hour=16,
                end_minute=0,
                activity_type=ActionType.WALK,
                location="town",
                sub_location="park",
                description="Walking in the park",
                specific_objects=["park_bench"],
                priority=6
            ))
        else:
            # Introverted - stay home
            activities.append(ScheduledActivity(
                start_hour=14,
                start_minute=0,
                end_hour=16,
                end_minute=0,
                activity_type=ActionType.USE_COMPUTER,
                location=home_location,
                sub_location="bedroom",
                description="Using computer",
                specific_objects=["desk", "computer"],
                priority=5
            ))
        
        return activities
    
    @staticmethod
    def _generate_evening_routine(
        personality: AgentPersonality,
        home_location: str,
        life_story: Optional[LifeStory]
    ) -> List[ScheduledActivity]:
        """Generate evening activities."""
        activities = []
        
        # Dinner preparation (6 PM - 7 PM)
        activities.append(ScheduledActivity(
            start_hour=18,
            start_minute=0,
            end_hour=18,
            end_minute=3,
            activity_type=ActionType.LOOK_IN_FRIDGE,
            location=home_location,
            sub_location="kitchen",
            description="Checking fridge for dinner ingredients",
            specific_objects=["fridge"],
            priority=7
        ))
        
        cook_duration = 40 if personality.conscientiousness > 0.6 else 25
        end_minute = 3 + cook_duration
        end_hour = 18
        if end_minute >= 60:
            end_hour += 1
            end_minute -= 60
        
        activities.append(ScheduledActivity(
            start_hour=18,
            start_minute=3,
            end_hour=end_hour,
            end_minute=end_minute,
            activity_type=ActionType.COOK,
            location=home_location,
            sub_location="kitchen",
            description="Cooking dinner",
            specific_objects=["stove", "oven", "counter", "cupboard"],
            priority=8
        ))
        
        activities.append(ScheduledActivity(
            start_hour=end_hour,
            start_minute=end_minute,
            end_hour=end_hour,
            end_minute=min(end_minute + 30, 59),
            activity_type=ActionType.EAT,
            location=home_location,
            sub_location="kitchen",
            description="Eating dinner",
            specific_objects=["table", "chair"],
            priority=8
        ))
        
        # Evening leisure (8 PM - 10 PM)
        if "socializing" in personality.favorite_activities and personality.extraversion > 0.6:
            activities.append(ScheduledActivity(
                start_hour=20,
                start_minute=0,
                end_hour=21,
                end_minute=30,
                activity_type=ActionType.CHAT,
                location="town",
                sub_location="cafe",
                description="Socializing at the cafe",
                specific_objects=["chair", "table"],
                priority=6
            ))
        elif "music" in personality.favorite_activities:
            activities.append(ScheduledActivity(
                start_hour=20,
                start_minute=0,
                end_hour=21,
                end_minute=30,
                activity_type=ActionType.LISTEN_MUSIC,
                location=home_location,
                sub_location="living room",
                description="Listening to music and relaxing",
                specific_objects=["couch", "stereo"],
                priority=5
            ))
        else:
            activities.append(ScheduledActivity(
                start_hour=20,
                start_minute=0,
                end_hour=21,
                end_minute=30,
                activity_type=ActionType.WATCH_TV,
                location=home_location,
                sub_location="living room",
                description="Watching TV",
                specific_objects=["couch", "tv"],
                priority=5
            ))
        
        return activities
    
    @staticmethod
    def _generate_night_routine(
        personality: AgentPersonality,
        home_location: str
    ) -> List[ScheduledActivity]:
        """Generate bedtime routine."""
        activities = []
        
        sleep_hour = personality.preferred_sleep_hour
        
        # Pre-bed activities (30 min before sleep)
        prep_hour = sleep_hour - 1 if sleep_hour > 0 else 23
        prep_minute = 30
        
        # Use toilet
        activities.append(ScheduledActivity(
            start_hour=prep_hour,
            start_minute=prep_minute,
            end_hour=prep_hour,
            end_minute=min(prep_minute + 5, 59),
            activity_type=ActionType.USE_TOILET,
            location=home_location,
            sub_location="bathroom",
            description="Using bathroom before bed",
            specific_objects=["toilet"],
            priority=8
        ))
        
        # Brush teeth
        activities.append(ScheduledActivity(
            start_hour=prep_hour,
            start_minute=min(prep_minute + 5, 59),
            end_hour=prep_hour,
            end_minute=min(prep_minute + 10, 59),
            activity_type=ActionType.BRUSH_TEETH,
            location=home_location,
            sub_location="bathroom",
            description="Brushing teeth",
            specific_objects=["sink", "mirror"],
            priority=8
        ))
        
        # Get into bed
        activities.append(ScheduledActivity(
            start_hour=prep_hour,
            start_minute=min(prep_minute + 10, 59),
            end_hour=sleep_hour,
            end_minute=0,
            activity_type=ActionType.READ_BOOK,
            location=home_location,
            sub_location="bedroom",
            description="Reading in bed before sleep",
            specific_objects=["bed", "lamp"],
            priority=6
        ))
        
        # Sleep
        wake_hour = personality.preferred_wake_hour
        activities.append(ScheduledActivity(
            start_hour=sleep_hour,
            start_minute=0,
            end_hour=wake_hour,
            end_minute=0,
            activity_type=ActionType.SLEEP,
            location=home_location,
            sub_location="bedroom",
            description="Sleeping",
            specific_objects=["bed"],
            priority=10
        ))
        
        return activities

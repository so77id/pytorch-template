
def get_dataset_params(dataset_name, cross=False):
    if dataset_name == "CK":
        if cross:
            return {
                "dataset_name": dataset_name,
                "type_load": "last",
                "n_labels": 6,
                "k": 5,
                "classes": [
                    "Anger",
                    "Disgust",
                    "Fear",
                    "Happiness",
                    "Sadness",
                    "Surprise"
                ]
            }
        else:
            return {
                "dataset_name": dataset_name,
                "type_load": "last",
                "n_labels": 7,
                "k": 5,
                "classes": [
                    "Anger",
                    "Contempt",
                    "Disgust",
                    "Fear",
                    "Happiness",
                    "Sadness",
                    "Surprise"
                ]
            }
    elif dataset_name == "MMI":
        return {
            "dataset_name": dataset_name,
            "type_load": "mid",
            "n_labels": 6,
            "k": 5,
            "classes": [
                "Anger",
                "Disgust",
                "Fear",
                "Happiness",
                "Sadness",
                "Surprise"
            ]
        }
    elif dataset_name == "OULU":
        return {
            "dataset_name": dataset_name,
            "type_load": "last",
            "n_labels": 6,
            "k": 5,
            "classes": [
                "Anger",
                "Disgust",
                "Fear",
                "Happiness",
                "Sadness",
                "Surprise"
            ]
        }
    elif dataset_name == "MUG":
        return {
            "dataset_name": dataset_name,
            "type_load": "mid",
            "n_labels": 6,
            "k": 5,
            "classes": [
                "Anger",
                "Disgust",
                "Fear",
                "Happiness",
                "Sadness",
                "Surprise"
            ]
        }
    elif dataset_name == "AFEW":
        return {
            "dataset_name": dataset_name,
            "type_load": "mid",
            "n_labels": 7,
            "k": 1,
            "classes": [
                "Angry",
                "Disgust",
                "Fear",
                "Happy",
                "Neutral",
                "Sad",
                "Surprise"
            ]
        }
    elif dataset_name == "UCF-101":
        return {
            "dataset_name": dataset_name,
            "type_load": "-",
            "n_labels": 101,
            "classes": [
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "Archery",
                "BabyCrawling",
                "BalanceBeam",
                "BandMarching",
                "BaseballPitch",
                "Basketball",
                "BasketballDunk",
                "BenchPress",
                "Biking",
                "Billiards",
                "BlowDryHair",
                "BlowingCandles",
                "BodyWeightSquats",
                "Bowling",
                "BoxingPunchingBag",
                "BoxingSpeedBag",
                "BreastStroke",
                "BrushingTeeth",
                "CleanAndJerk",
                "CliffDiving",
                "CricketBowling",
                "CricketShot",
                "CuttingInKitchen",
                "Diving",
                "Drumming",
                "Fencing",
                "FieldHockeyPenalty",
                "FloorGymnastics",
                "FrisbeeCatch",
                "FrontCrawl",
                "GolfSwing",
                "Haircut",
                "Hammering",
                "HammerThrow",
                "HandstandPushups",
                "HandstandWalking",
                "HeadMassage",
                "HighJump",
                "HorseRace",
                "HorseRiding",
                "HulaHoop",
                "IceDancing",
                "JavelinThrow",
                "JugglingBalls",
                "JumpingJack",
                "JumpRope",
                "Kayaking",
                "Knitting",
                "LongJump",
                "Lunges",
                "MilitaryParade",
                "Mixing",
                "MoppingFloor",
                "Nunchucks",
                "ParallelBars",
                "PizzaTossing",
                "PlayingCello",
                "PlayingDaf",
                "PlayingDhol",
                "PlayingFlute",
                "PlayingGuitar",
                "PlayingPiano",
                "PlayingSitar",
                "PlayingTabla",
                "PlayingViolin",
                "PoleVault",
                "PommelHorse",
                "PullUps",
                "Punch",
                "PushUps",
                "Rafting",
                "RockClimbingIndoor",
                "RopeClimbing",
                "Rowing",
                "SalsaSpin",
                "ShavingBeard",
                "Shotput",
                "SkateBoarding",
                "Skiing",
                "Skijet",
                "SkyDiving",
                "SoccerJuggling",
                "SoccerPenalty",
                "StillRings",
                "SumoWrestling",
                "Surfing",
                "Swing",
                "TableTennisShot",
                "TaiChi",
                "TennisSwing",
                "ThrowDiscus",
                "TrampolineJumping",
                "Typing",
                "UnevenBars",
                "VolleyballSpiking",
                "WalkingWithDog",
                "WallPushups",
                "WritingOnBoard",
                "YoYo"
            ]
        }

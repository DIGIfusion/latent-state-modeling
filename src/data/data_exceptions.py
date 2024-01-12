
class NotDesiredShot(Exception): 
    def __init__(self, reason: str, shotno: str) -> None:
        self.shotno = shotno
        self.reason = reason
        self.message = f'{shotno} not desired because {reason}'
        super().__init__(reason, shotno)

class RawPulseDictErrorMissingInformation(Exception): 
    def __init__(self, reason: str, shotno: str) -> None: 
        self.shotno = shotno
        self.reason = reason
        self.message = f'{shotno} not saved because {reason} did not exist'
        super().__init__(reason, shotno)

class ShortPulse(Exception): 
    def __init__(self, shotno: str, total_time: float) -> None: 
        self.shotno = shotno
        self.total_time = total_time
        self.message = f'{shotno} not saved because only {total_time}s after making array format'
        super().__init__(total_time, shotno)

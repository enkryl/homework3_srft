from abc import ABC, abstractmethod
from typing import Optional, List

from base.verifier import Verifier
from base.data import Data


class Env(ABC):
    """
    Base class for game
    @param name: name of the game
    @param verifier: class of the verifier
    """

    def __init__(self, name: str, verifier: Verifier):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
    ) -> List[Data]:
        """
        Generate game questions and answers
        @param num_of_questions: int
        @param max_attempts: int
        @return: list of Data
        """
        raise NotImplementedError("Game.generate() is not implemented")

    def verify(self, data: Data, test_solution: str) -> bool:
        """
        Verify whether the test solution is consistent with the answer of the game data
        @param data: Data
        @param test_solution: str
        @return: bool
        """
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str) -> str:
        """
        Extract the answer from the test solution
        @param test_solution: str
        @return: str
        """
        raise NotImplementedError("Game.extract_answer() is not implemented")
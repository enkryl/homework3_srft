from abc import ABC, abstractmethod
from base.data import Data


class Verifier(ABC):
    """
    Base class for verifier
    """

    def __init__(self):
        pass

    @abstractmethod
    def verify(self, data: Data, test_answer: str) -> bool:
        """
        Verify whether the test answer is consistent with the gold answer
        @param data: Data
        @param test_answer: str
        @return: bool
        """
        raise NotImplementedError("Verifier.verify() is not implemented")

    @abstractmethod
    def extract_answer(self, test_solution: str) -> str:
        """
        Extract the answer from the test solution
        @param test_solution: str
        @return: str
        """
        raise NotImplementedError("Verifier.extract_answer() is not implemented")
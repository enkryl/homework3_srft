import json
from typing import Optional, Dict, Any, List


class Data:
    """
    Data class for game/corpus
    @param question: question of the game/corpus
    @param answer: answer of the game/corpus
    @param difficulty: difficulty of the game/corpus, from 1 to 10
    """

    def __init__(
        self,
        question: str,
        answer: str,
        difficulty: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.metadata = metadata
        self.gpt_response = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response,
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)

    @classmethod
    def from_json_str(cls, json_str: str) -> "Data":
        json_data = json.loads(json_str)
        return cls(**json_data)

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> "Data":
        instance = cls(**json_dict)
        if "gpt_response" in json_dict:
            instance.gpt_response = json_dict["gpt_response"]
        return instance

    @classmethod
    def from_jsonl_file(cls, file_path: str) -> List["Data"]:
        data_list: List["Data"] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                json_data = json.loads(line)
                instance = cls(**json_data)
                if "gpt_response" in json_data:
                    instance.gpt_response = json_data["gpt_response"]
                data_list.append(instance)
        return data_list
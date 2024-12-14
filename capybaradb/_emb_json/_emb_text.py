from typing import Optional, Dict


class EmbText:
    SUPPORTED_EMB_MODELS = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "ada v2",
    ]

    def __init__(self, text: str, emb_model: Optional[str] = None):
        if not self.is_valid_text(text):
            raise ValueError("Invalid text: must be a non-empty string.")
        if emb_model and not self.is_valid_emb_model(emb_model):
            raise ValueError(f"Invalid embedding model: {emb_model} is not supported.")

        self.text = text
        self.emb_model = emb_model

    @staticmethod
    def is_valid_text(text: str) -> bool:
        return isinstance(text, str) and text.strip() != ""

    @classmethod
    def is_valid_emb_model(cls, emb_model: str) -> bool:
        return emb_model in cls.SUPPORTED_EMB_MODELS

    def to_json(self) -> Dict[str, Dict[str, str]]:
        """
        Convert the EmbText instance to a JSON-serializable dictionary.
        """
        data = {"@embText": {"text": self.text}}
        if self.emb_model:
            data["@embText"]["emb_model"] = self.emb_model
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Dict[str, str]]) -> "EmbText":
        """
        Create an EmbText instance from a JSON-serializable dictionary.
        """
        emb_text_data = data.get("@embText", {})
        text = emb_text_data.get("text")
        emb_model = emb_text_data.get("emb_model")
        return cls(text, emb_model)

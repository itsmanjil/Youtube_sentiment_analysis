"""
Centralized configuration and path management.

This module provides a single source of truth for all paths and configuration
values used throughout the sentiment analysis system. This ensures consistency
and makes the codebase easier to maintain.
"""

import os
from pathlib import Path
from typing import Optional, Union


class Config:
    """
    Centralized configuration for the sentiment analysis system.

    This class provides standardized paths for models, data, and other
    resources. All paths are resolved relative to the backend directory.

    Attributes
    ----------
    BACKEND_DIR : Path
        Root directory of the backend package.
    MODELS_DIR : Path
        Directory containing trained model artifacts.
    DATA_DIR : Path
        Directory for training and evaluation data.
    EMBEDDINGS_DIR : Path
        Directory for pre-trained word embeddings.
    OUTPUT_DIR : Path
        Directory for training outputs and experiment results.
    CONFIGS_DIR : Path
        Directory for configuration files.
    FILES_DIR : Path
        Directory for auxiliary files (contractions, etc.).

    Examples
    --------
    >>> from src.utils.config import Config
    >>> Config.MODELS_DIR
    PosixPath('/path/to/backend/models')

    >>> Config.get_model_path('logreg')
    PosixPath('/path/to/backend/models/logreg/model.sav')
    """

    # Base directories
    BACKEND_DIR: Path = Path(__file__).resolve().parents[2]
    SRC_DIR: Path = BACKEND_DIR / "src"

    # Resource directories
    MODELS_DIR: Path = BACKEND_DIR / "models"
    DATA_DIR: Path = BACKEND_DIR / "data"
    EMBEDDINGS_DIR: Path = BACKEND_DIR / "embeddings"
    OUTPUT_DIR: Path = BACKEND_DIR / "output"
    CONFIGS_DIR: Path = BACKEND_DIR / "configs"
    FILES_DIR: Path = BACKEND_DIR / "files"
    RESEARCH_DIR: Path = BACKEND_DIR / "research"

    # Model subdirectories
    MODEL_PATHS = {
        "tfidf": {
            "model": MODELS_DIR / "tfidf" / "model.sav",
            "vectorizer": MODELS_DIR / "tfidf" / "tfidfVectorizer.pickle",
        },
        "logreg": {
            "model": MODELS_DIR / "logreg" / "model.sav",
            "vectorizer": MODELS_DIR / "logreg" / "tfidfVectorizer.pickle",
        },
        "svm": {
            "model": MODELS_DIR / "svm" / "model.sav",
            "vectorizer": MODELS_DIR / "svm" / "tfidfVectorizer.pickle",
        },
        "hybrid_dl": {
            "model": MODELS_DIR / "hybrid_dl" / "hybrid_v1.pt",
            "vocab": MODELS_DIR / "hybrid_dl" / "vocab.pkl",
            "metadata": MODELS_DIR / "hybrid_dl" / "metadata.json",
        },
        "meta_learner": {
            "model": MODELS_DIR / "meta_learner.pkl",
        },
        "bert": {
            "model": MODELS_DIR / "transformers" / "bert",
        },
    }

    # Auxiliary files
    CONTRACTIONS_FILE: Path = FILES_DIR / "contractions.json"
    NEGATIONS_FILE: Path = FILES_DIR / "negations.json"

    @classmethod
    def get_model_path(
        cls, model_type: str, component: str = "model"
    ) -> Optional[Path]:
        """
        Get the path for a specific model component.

        Parameters
        ----------
        model_type : str
            Type of model ('tfidf', 'logreg', 'svm', 'hybrid_dl', 'meta_learner', 'bert').
        component : str, optional
            Component to retrieve ('model', 'vectorizer', 'vocab', 'metadata').
            Default is 'model'.

        Returns
        -------
        Optional[Path]
            Path to the requested component, or None if not found.

        Examples
        --------
        >>> Config.get_model_path('logreg', 'model')
        PosixPath('/path/to/backend/models/logreg/model.sav')

        >>> Config.get_model_path('logreg', 'vectorizer')
        PosixPath('/path/to/backend/models/logreg/tfidfVectorizer.pickle')
        """
        model_type = model_type.lower().strip()
        if model_type not in cls.MODEL_PATHS:
            return None
        return cls.MODEL_PATHS[model_type].get(component)

    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create all required directories if they don't exist.

        This is useful for initial setup or when running in a new environment.
        """
        directories = [
            cls.MODELS_DIR,
            cls.DATA_DIR,
            cls.EMBEDDINGS_DIR,
            cls.OUTPUT_DIR,
            cls.CONFIGS_DIR,
            cls.FILES_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def get_model_path(path_value: Union[str, Path]) -> Path:
    """
    Resolve a model path to an absolute path.

    If the path is relative, it's resolved relative to the backend directory.

    Parameters
    ----------
    path_value : Union[str, Path]
        Path to resolve. Can be absolute or relative.

    Returns
    -------
    Path
        Resolved absolute path.

    Examples
    --------
    >>> get_model_path('./models/logreg/model.sav')
    PosixPath('/path/to/backend/models/logreg/model.sav')

    >>> get_model_path('/absolute/path/to/model.sav')
    PosixPath('/absolute/path/to/model.sav')
    """
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (Config.BACKEND_DIR / path_obj).resolve()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable with optional default.

    Parameters
    ----------
    key : str
        Environment variable name.
    default : Optional[str], optional
        Default value if not found. Default is None.

    Returns
    -------
    Optional[str]
        Environment variable value or default.
    """
    return os.environ.get(key, default)

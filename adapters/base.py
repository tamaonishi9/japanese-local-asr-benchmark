from __future__ import annotations

from abc import ABC, abstractmethod

from engine_result import EngineResult
from source_audio import SourceAudio


class BaseAdapter(ABC):
    @abstractmethod
    def prepare(self) -> None:
        """モデルを読み込み推論可能な状態にする。model_load_seconds を計測する。"""
        ...

    @abstractmethod
    def run(self, source_audio: SourceAudio) -> EngineResult:
        """source audio を受け取り推論を実行して EngineResult を返す。"""
        ...

    def release(self) -> None:
        """モデルを解放してメモリを回収する。解放処理が不要な adapter は上書き不要。"""
        pass

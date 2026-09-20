"""Ensamblador de vídeo: combina MP4 + audio + subtítulos con ffmpeg."""

import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - dependencia declarada en requirements.txt
    imageio_ffmpeg = None

from engines.base_engine import EngineError

logger = logging.getLogger(__name__)

# Raíz del proyecto (C:\Imperator); output/ cuelga de aquí.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Mapeo del estilo de subtítulos (claves internas de la UI) a ASS.
_POSITION_ALIGNMENT = {"bottom": 2, "middle": 5, "top": 8}
_COLOR_HEX = {
    "white": "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "green": "&H0000FF00",
}
_DEFAULT_STYLE = {
    "fontsize": 24,
    "color": "white",
    "position": "bottom",
    "outline": 2,
}


class VideoAssembler:
    """Servicio que ensambla un MP4 final (vídeo + audio + subtítulos).

    Usa el binario ffmpeg de imageio_ffmpeg. No es un :class:`Engine`: es una
    utilidad local invocada por la tarea Ensamblador.
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self._output_dir = (
            Path(output_dir) if output_dir is not None else PROJECT_ROOT / "output"
        )

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @staticmethod
    def _ffmpeg() -> str:
        if imageio_ffmpeg is None:
            raise EngineError(
                "imageio-ffmpeg no está instalado. "
                "Instálalo con: pip install imageio-ffmpeg>=0.6.0"
            )
        return imageio_ffmpeg.get_ffmpeg_exe()

    def probe(self, path: str | Path) -> dict:
        """Devuelve ``{duration, width, height}`` del fichero (None si no se detecta).

        Se apoya en la salida stderr de ``ffmpeg -i`` (imageio-ffmpeg no incluye
        ffprobe), para no añadir dependencias nuevas.
        """
        ffmpeg = self._ffmpeg()
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(path)],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise EngineError(f"No se pudo ejecutar ffmpeg para inspeccionar: {exc}") from exc

        stderr = proc.stderr or ""
        info: dict = {"duration": None, "width": None, "height": None}

        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
        if match:
            info["duration"] = (
                int(match.group(1)) * 3600
                + int(match.group(2)) * 60
                + float(match.group(3))
            )

        match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", stderr)
        if match:
            info["width"] = int(match.group(1))
            info["height"] = int(match.group(2))

        return info

    def assemble(
        self,
        video_path: str | Path,
        audio_path: str | Path | None = None,
        srt_path: str | Path | None = None,
        subtitle_style: dict | None = None,
        music_path: str | Path | None = None,
        music_volume: float = 0.3,
        voice_volume: float = 1.0,
        ducking: bool = True,
    ) -> Path:
        """Ensambla el MP4 final y devuelve su ruta.

        Combina hasta cuatro entradas: un MP4 (obligatorio), una voz (opcional),
        subtítulos SRT (opcional) y una pista de música de fondo (opcional). Si
        hay SRT, genera un ASS temporal con el estilo indicado y lo quema en el
        vídeo. Con música, la mezcla voz+música (ducking opcional vía
        ``sidechaincompress``) se hace con ``-filter_complex``.
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            raise EngineError(f"El vídeo no existe: {video_path}")

        audio_path = Path(audio_path) if audio_path is not None else None
        if audio_path is not None and not audio_path.is_file():
            raise EngineError(f"El audio no existe: {audio_path}")

        srt_path = Path(srt_path) if srt_path is not None else None
        if srt_path is not None and not srt_path.is_file():
            raise EngineError(f"El SRT no existe: {srt_path}")

        music_path = Path(music_path) if music_path is not None else None
        if music_path is not None and not music_path.is_file():
            raise EngineError(f"La música no existe: {music_path}")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / self._output_name()

        ass_path: Path | None = None
        if srt_path is not None:
            info = self.probe(video_path)
            ass_path = self._build_ass(
                srt_path,
                subtitle_style,
                video_width=info.get("width"),
                video_height=info.get("height"),
            )

        cmd = self._build_command(
            video_path,
            audio_path,
            ass_path,
            output_path,
            music_path,
            music_volume,
            voice_volume,
            ducking,
        )
        try:
            self._run(cmd)
        finally:
            if ass_path is not None:
                ass_path.unlink(missing_ok=True)

        logger.info("VideoAssembler: MP4 final en '%s'.", output_path)
        return output_path

    def _build_command(
        self,
        video_path: Path,
        audio_path: Path | None,
        ass_path: Path | None,
        output_path: Path,
        music_path: Path | None,
        music_volume: float,
        voice_volume: float,
        ducking: bool,
    ) -> list[str]:
        """Construye el comando ffmpeg según el caso (1-8)."""
        if music_path is not None:
            return self._build_music_command(
                video_path,
                audio_path,
                ass_path,
                output_path,
                music_path,
                music_volume,
                voice_volume,
                ducking,
            )

        # Sin música: casos 1-4.
        ffmpeg = self._ffmpeg()
        has_voice = audio_path is not None
        burn = ass_path is not None

        cmd = [ffmpeg, "-y", "-i", str(video_path)]
        if has_voice:
            cmd += ["-i", str(audio_path)]

        if not has_voice and not burn:
            # Caso 1: solo vídeo -> copia de streams.
            cmd += ["-c", "copy"]
        elif has_voice and not burn:
            # Caso 2: vídeo + voz -> muxing (vídeo copiado, audio AAC).
            cmd += [
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
            ]
        elif burn:
            # Caso 3/4: re-codificar vídeo y quemar subtítulos.
            cmd += [
                "-vf", f"subtitles='{self._filter_path(ass_path)}'",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            ]
            if has_voice:
                cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
            else:
                cmd += ["-c:a", "copy"]

        cmd += ["-movflags", "+faststart"]
        cmd.append(str(output_path))
        return cmd

    def _build_music_command(
        self,
        video_path: Path,
        audio_path: Path | None,
        ass_path: Path | None,
        output_path: Path,
        music_path: Path,
        music_volume: float,
        voice_volume: float,
        ducking: bool,
    ) -> list[str]:
        """Casos 5-8: música de fondo (ducking opcional) vía ``-filter_complex``.

        Cuando además hay subtítulos, el filtro ``subtitles`` se mete dentro del
        mismo ``-filter_complex`` (no puede combinarse ``-vf`` con ``-filter_complex``).
        """
        ffmpeg = self._ffmpeg()
        has_voice = audio_path is not None
        burn = ass_path is not None

        cmd = [ffmpeg, "-y", "-i", str(video_path)]
        if has_voice:
            cmd += ["-i", str(audio_path)]
        # -stream_loop -1 se aplica al input siguiente (la música) para loopearla.
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]

        voice_idx = 1 if has_voice else None
        music_idx = 2 if has_voice else 1

        filter_parts: list[str] = []
        if burn:
            vf = f"subtitles='{self._filter_path(ass_path)}'"
            filter_parts.append(f"[0:v]{vf}[vout]")
        filter_parts.append(
            self._audio_filter_complex(
                has_voice, voice_idx, music_idx, music_volume, voice_volume, ducking
            )
        )
        cmd += ["-filter_complex", ";".join(filter_parts)]

        if burn:
            cmd += ["-map", "[vout]", "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-map", "0:v:0", "-c:v", "copy"]

        cmd += ["-map", "[mix]", "-c:a", "aac"]
        cmd += ["-shortest", "-movflags", "+faststart"]
        cmd.append(str(output_path))
        return cmd

    @staticmethod
    def _audio_filter_complex(
        has_voice: bool,
        voice_idx: int | None,
        music_idx: int,
        music_volume: float,
        voice_volume: float,
        ducking: bool,
    ) -> str:
        """Devuelve la rama de audio del ``-filter_complex`` (produce ``[mix]``).

        Todas las ramas normalizan formato con ``aresample=48000`` y
        ``aformat=channel_layouts=stereo`` (incluye el upmix mono->estéreo).
        """
        mv = f"{music_volume:.3f}"
        vv = f"{voice_volume:.3f}"

        if not has_voice:
            # Caso 7/8: solo música de fondo.
            return (
                f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
                f"volume={mv}[mix]"
            )

        # Caso 5/6: voz + música (ducking opcional).
        if ducking:
            voice_chain = (
                f"[{voice_idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
                f"volume={vv},asplit=2[v1][vsc]"
            )
        else:
            voice_chain = (
                f"[{voice_idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
                f"volume={vv}[v1]"
            )
        music_chain = (
            f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"volume={mv}[music]"
        )
        if ducking:
            mid = (
                "[music][vsc]sidechaincompress="
                "threshold=0.05:ratio=8:attack=10:release=250[md]"
            )
            mix_in = "[md]"
        else:
            mid = None
            mix_in = "[music]"

        amix = (
            f"{mix_in}[v1]amix=inputs=2:duration=first:dropout_transition=3:"
            f"normalize=0[mix]"
        )

        parts = [voice_chain, music_chain]
        if mid:
            parts.append(mid)
        parts.append(amix)
        return ";".join(parts)

    @staticmethod
    def _run(cmd: list[str]) -> None:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            raise EngineError(f"No se pudo ejecutar ffmpeg: {exc}") from exc
        if proc.returncode != 0:
            lines = (proc.stderr or "").strip().splitlines()
            detail = "\n".join(lines[-15:]) or "error desconocido"
            raise EngineError(f"ffmpeg falló:\n{detail}")

    def _output_name(self) -> str:
        return f"plataformaia_{datetime.now():%Y%m%d_%H%M%S_%f}_ensamblado.mp4"

    # ---- Subtítulos (SRT -> ASS) ---------------------------------

    def _build_ass(
        self,
        srt_path: Path,
        style: dict | None,
        video_width: int | None = None,
        video_height: int | None = None,
    ) -> Path:
        """Genera un ASS temporal con el estilo pedido y devuelve su ruta.

        ``PlayRes`` se ajusta a la resolución del vídeo para que ``fontsize`` y
        ``MarginV`` se interpreten en píxeles absolutos (p. ej. 24 px, 60 px).
        """
        merged = {**_DEFAULT_STYLE, **(style or {})}
        fontsize = int(merged["fontsize"])
        color = _COLOR_HEX.get(merged.get("color"), _COLOR_HEX["white"])
        alignment = _POSITION_ALIGNMENT.get(merged.get("position"), 2)
        outline = int(merged.get("outline", 2))

        events = self._parse_srt(srt_path)

        play_x = video_width or 384
        play_y = video_height or 288

        header = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {play_x}",
            f"PlayResY: {play_y}",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            (
                f"Style: Default,Arial,{fontsize},{color},&H000000FF,"
                f"&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{outline},1,"
                f"{alignment},20,20,60,1"
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text",
        ]
        body = [
            f"Dialogue: 0,{self._ass_time(start)},{self._ass_time(end)},"
            f"Default,,0,0,0,,{text}"
            for start, end, text in events
        ]

        ass_path = srt_path.with_suffix(".ass")
        ass_path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
        return ass_path

    @staticmethod
    def _parse_srt(srt_path: Path) -> list[tuple[float, float, str]]:
        """Parsea un SRT (el generado por WhisperEngine) en (inicio, fin, texto)."""
        content = srt_path.read_text(encoding="utf-8")
        blocks = re.split(r"\n\s*\n", content.strip())
        events: list[tuple[float, float, str]] = []
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if len(lines) < 2:
                continue
            time_line = next((ln for ln in lines if "-->" in ln), None)
            if time_line is None:
                continue
            match = re.match(
                r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)",
                time_line,
            )
            if not match:
                continue
            start = VideoAssembler._time_to_seconds(*match.groups()[:4])
            end = VideoAssembler._time_to_seconds(*match.groups()[4:])
            idx = lines.index(time_line)
            text = "\\N".join(lines[idx + 1:])
            events.append((start, end, text))
        return events

    @staticmethod
    def _time_to_seconds(h: str, m: str, s: str, frac: str) -> float:
        return int(h) * 3600 + int(m) * 60 + int(s) + int(frac.ljust(3, "0")) / 1000

    @staticmethod
    def _ass_time(seconds: float) -> str:
        """Formatea segundos como ``H:MM:SS.cc`` (centisegundos, estándar ASS)."""
        total_cs = max(0, int(round(seconds * 100)))
        hours, total_cs = divmod(total_cs, 360_000)
        minutes, total_cs = divmod(total_cs, 6_000)
        secs, cs = divmod(total_cs, 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"

    @staticmethod
    def _filter_path(path: Path) -> str:
        """Escapa una ruta de Windows para usarla dentro de un filtergraph."""
        return str(path).replace("\\", "/").replace(":", "\\:")

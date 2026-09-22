"""Genera la narración épica del guion de Basajaun con Piper TTS (voces locales).

Usa el motor propio de Imperator (engines.piper_tts.PiperTTSEngine) para sintetizar
dos versiones:

- Castellano (es_ES-davefx-medium)
- Euskera  (eu_ES-antton-medium)

Salida: C:\\Imperator\\output\\basajaun_narracion_es.wav y ..._eu.wav
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engines.piper_tts import PiperTTSEngine  # noqa: E402

# --- Guion de Basajaun ------------------------------------------------

GUION_ES = (
    "En lo más profundo de los bosques del norte, donde los robles rozan el cielo "
    "y la niebla abraza las montañas, habita Basajaun, el Señor del Bosque. "
    "Alto como un haya centenaria, con el cuerpo cubierto de un vello espeso "
    "y los ojos encendidos como brasas, camina en silencio entre las sombras. "
    "Los pastores cuentan que protege los rebaños, y que avisa con su bramido "
    "cuando la tormenta amenaza en la montaña. "
    "Dicen las leyendas que fue él quien enseñó a los hombres a sembrar el trigo, "
    "a forjar el hierro y a mover la piedra del molino. "
    "No temas, caminante: si respetas el bosque y a sus criaturas, "
    "Basajaun velará por tu senda. "
    "Pero si alzas la mano contra un árbol sagrado, "
    "escucharás su rugido resonar entre los valles. "
    "Esta es la historia del guardián eterno de Euskal Herria."
)

GUION_EU = (
    "Basajaun, basoaren jauna, mendietako haitzuloetan bizi da. "
    "Erraldoi iletsua da, eta begiak sugarra bezala dizdiz egiten dute. "
    "Artzainei ekaitza iragartzen die, eta artaldeak zaintzen ditu gauez. "
    "Kondairak dio gizakiei garia ereiten, burdina lantzen eta errotaren harria "
    "mugitzen irakatsi ziela. "
    "Errespetatu basoa eta bere izakiak, eta Basajaunek zure bidea zainduko du. "
    "Hau da Euskal Herriko zaindariaren istorioa."
)


def main() -> None:
    engine = PiperTTSEngine(
        voices_dir="voices",
        default_voice="es_ES-davefx-medium",
        output_dir="output",
    )

    out_es = engine.synthesize(
        GUION_ES,
        voice="es_ES-davefx-medium",
        length_scale=1.05,
        output_path=PROJECT_ROOT / "output" / "basajaun_narracion_es.wav",
    )
    print(f"ES -> {out_es}")

    out_eu = engine.synthesize(
        GUION_EU,
        voice="eu_ES-antton-medium",
        length_scale=1.05,
        output_path=PROJECT_ROOT / "output" / "basajaun_narracion_eu.wav",
    )
    print(f"EU -> {out_eu}")


if __name__ == "__main__":
    main()

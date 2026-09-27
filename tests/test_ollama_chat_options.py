"""Tests del payload de ``OllamaEngine.chat`` y de las respuestas vacías.

Contexto del bug: con la ventana de contexto llena (``truncated``) o el
presupuesto de salida gastado pensando, Ollama responde **HTTP 200** con
``message.content`` vacío. El chat lo pintaba como un turno en blanco
(``Ollama:``) y guardaba un mensaje de asistente vacío en el historial.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import ollama as ollama_mod  # noqa: E402
from engines.base_engine import EngineError  # noqa: E402
from engines.ollama import OllamaEngine  # noqa: E402


def _response(payload: dict) -> mock.Mock:
    resp = mock.Mock()
    resp.json.return_value = payload
    return resp


class ChatPayloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = OllamaEngine()

    def _chat(self, payload: dict) -> tuple[str, dict]:
        with mock.patch.object(
            ollama_mod.requests, "post", return_value=_response(payload)
        ) as post:
            result = self.engine.chat([{"role": "user", "content": "hola"}])
        return result, post.call_args.kwargs["json"]

    def test_pide_ventana_de_contexto_y_tokens_de_salida(self) -> None:
        contenido, enviado = self._chat(
            {"message": {"content": "hola"}, "done_reason": "stop"}
        )
        self.assertEqual(contenido, "hola")
        opciones = enviado["options"]
        self.assertEqual(opciones["num_ctx"], OllamaEngine._NUM_CTX)
        self.assertEqual(opciones["num_predict"], OllamaEngine._NUM_PREDICT)

    def test_los_limites_se_pueden_configurar(self) -> None:
        engine = OllamaEngine(num_ctx=32768, num_predict=4096)
        with mock.patch.object(
            ollama_mod.requests,
            "post",
            return_value=_response({"message": {"content": "ok"}}),
        ) as post:
            engine.chat([{"role": "user", "content": "hola"}])
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_ctx"], 32768)
        self.assertEqual(
            post.call_args.kwargs["json"]["options"]["num_predict"], 4096
        )

    def test_recorta_espacios_del_contenido(self) -> None:
        contenido, _ = self._chat({"message": {"content": "  hola  "}})
        self.assertEqual(contenido, "hola")

    def test_por_defecto_no_limita_los_tokens_de_salida(self) -> None:
        # Un tope bajo cortaba las respuestas a media frase: el techo real es la
        # ventana de contexto.
        self.assertEqual(OllamaEngine._NUM_PREDICT, -1)
        _, enviado = self._chat({"message": {"content": "hola"}})
        self.assertEqual(enviado["options"]["num_predict"], -1)

    def test_respuesta_cortada_avisa_en_el_propio_texto(self) -> None:
        contenido, _ = self._chat(
            {"message": {"content": "texto a medias"}, "done_reason": "length"}
        )
        self.assertTrue(contenido.startswith("texto a medias"))
        self.assertTrue(contenido.endswith(OllamaEngine.TRUNCATED_NOTE))
        self.assertIn("respuesta incompleta", contenido)

    def test_respuesta_completa_no_lleva_aviso(self) -> None:
        contenido, _ = self._chat(
            {"message": {"content": "respuesta entera"}, "done_reason": "stop"}
        )
        self.assertEqual(contenido, "respuesta entera")


class EmptyReplyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = OllamaEngine()

    def _error(self, payload: dict) -> EngineError:
        with mock.patch.object(
            ollama_mod.requests, "post", return_value=_response(payload)
        ):
            with self.assertRaises(EngineError) as ctx:
                self.engine.chat([{"role": "user", "content": "hola"}])
        return ctx.exception

    def test_contenido_vacio_lanza_error_con_pista(self) -> None:
        error = self._error({"message": {"content": ""}, "done_reason": "stop"})
        mensaje = str(error)
        self.assertIn("la respuesta llegó vacía", mensaje)
        self.assertIn("num_ctx", mensaje)
        self.assertIn("num_predict", mensaje)

    def test_truncado_por_longitud_lo_explica(self) -> None:
        # Este es el caso del log del servidor: n_tokens agotados, truncated=1.
        error = self._error(
            {"message": {"content": ""}, "done_reason": "length"}
        )
        self.assertIn("se agotó el espacio", str(error))

    def test_respuesta_con_solo_razonamiento_lo_explica(self) -> None:
        error = self._error(
            {
                "message": {"content": "", "thinking": "pensando mucho..."},
                "done_reason": "stop",
            }
        )
        self.assertIn("pensando", str(error))

    def test_sin_mensaje_en_la_respuesta_tambien_lanza(self) -> None:
        error = self._error({"done_reason": "stop"})
        self.assertIn("vacía", str(error))


if __name__ == "__main__":
    unittest.main()

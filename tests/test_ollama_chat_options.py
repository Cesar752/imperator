"""Tests del payload de ``OllamaEngine.chat`` y de las respuestas vacías.

Contexto del bug: con la ventana de contexto llena (``truncated``) o el
presupuesto de salida gastado pensando, Ollama responde **HTTP 200** con
``message.content`` vacío. El chat lo pintaba como un turno en blanco
(``Ollama:``) y guardaba un mensaje de asistente vacío en el historial.

Aquí vive también el mapeo del HTTP 400: el texto de ``raise_for_status`` ("400
Client Error: Bad Request for url: ...") ocultaba el motivo real, que viaja en el
cuerpo (``{"error": "request (4135 tokens) exceeds the available context size
(4096 tokens), try increasing it"}``).
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import ollama as ollama_mod  # noqa: E402
from engines import ollama_api  # noqa: E402
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

    def test_por_defecto_no_manda_think(self) -> None:
        # Sin configurar, se deja el comportamiento del modelo (no se envía nada).
        _, enviado = self._chat({"message": {"content": "hola"}})
        self.assertNotIn("think", enviado)
        self.assertNotIn(
            OllamaEngine.NO_THINK_HINT, enviado["messages"][-1]["content"]
        )

    def test_think_false_apaga_el_razonamiento(self) -> None:
        # Los qwen3.x gastaban la ventana «pensando» y el Cuaderno/Crónica salían
        # vacíos («se agotó el espacio de la respuesta»): con think=False se pide
        # el campo y, además, se añade el marcador /no_think por si se ignora.
        engine = OllamaEngine(think=False)
        with mock.patch.object(
            ollama_mod.requests,
            "post",
            return_value=_response({"message": {"content": "ok"}}),
        ) as post:
            engine.chat([{"role": "user", "content": "hola"}])
        enviado = post.call_args.kwargs["json"]
        self.assertIs(enviado["think"], False)
        self.assertIn(OllamaEngine.NO_THINK_HINT, enviado["messages"][-1]["content"])

    def test_think_false_no_duplica_el_marcador(self) -> None:
        engine = OllamaEngine(think=False)
        with mock.patch.object(
            ollama_mod.requests,
            "post",
            return_value=_response({"message": {"content": "ok"}}),
        ) as post:
            engine.chat([{"role": "user", "content": f"hola {OllamaEngine.NO_THINK_HINT}"}])
        contenido = post.call_args.kwargs["json"]["messages"][-1]["content"]
        self.assertEqual(contenido.count(OllamaEngine.NO_THINK_HINT), 1)

    def test_el_gestor_deja_configurar_think(self) -> None:
        from core.engine_manager import EngineManager

        self.assertIs(
            EngineManager._build("ollama", {"parametros": {"think": False}})._think,
            False,
        )
        self.assertIsNone(
            EngineManager._build("ollama", {"parametros": {}})._think
        )

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


class ContextErrorTest(unittest.TestCase):
    """El 400 por contexto se explica: el prompt no cabía, y qué hacer."""

    _DETALLE = (
        "request (4135 tokens) exceeds the available context size (4096 tokens), "
        "try increasing it"
    )

    def _chat_error(self, payload: dict, status_code: int = 400) -> EngineError:
        respuesta = requests.models.Response()
        respuesta.status_code = status_code
        respuesta._content = json.dumps(payload).encode("utf-8")
        resp = mock.Mock()
        resp.raise_for_status.side_effect = requests.HTTPError(
            str(status_code), response=respuesta
        )
        with mock.patch.object(ollama_mod.requests, "post", return_value=resp):
            with self.assertRaises(EngineError) as ctx:
                OllamaEngine().chat([{"role": "user", "content": "hola"}])
        return ctx.exception

    def test_el_contexto_excedido_se_explica(self) -> None:
        mensaje = str(self._chat_error({"error": self._DETALLE}))
        self.assertIn("no cabe en el contexto", mensaje)
        self.assertIn(f"num_ctx={OllamaEngine._NUM_CTX}", mensaje)
        self.assertIn("4135", mensaje)
        self.assertIn("OLLAMA_CONTEXT_LENGTH", mensaje)

    def test_un_400_sin_detalle_no_pierde_el_codigo(self) -> None:
        self.assertIn("HTTP 400", str(self._chat_error({})))

    def test_un_400_con_cuerpo_no_json_usa_el_texto(self) -> None:
        respuesta = requests.models.Response()
        respuesta.status_code = 400
        respuesta._content = b"bad request"
        resp = mock.Mock()
        resp.raise_for_status.side_effect = requests.HTTPError(
            "400", response=respuesta
        )
        with mock.patch.object(ollama_mod.requests, "post", return_value=resp):
            with self.assertRaises(EngineError) as ctx:
                OllamaEngine().chat([{"role": "user", "content": "hola"}])
        self.assertIn("bad request", str(ctx.exception))

    def test_un_error_http_sin_cuerpo_no_revienta(self) -> None:
        # ``requests`` puede elevar HTTPError sin respuesta asociada.
        resp = mock.Mock()
        resp.raise_for_status.side_effect = requests.HTTPError("400")
        with mock.patch.object(ollama_mod.requests, "post", return_value=resp):
            with self.assertRaises(EngineError) as ctx:
                OllamaEngine().chat([{"role": "user", "content": "hola"}])
        self.assertIn("desconocido", str(ctx.exception))


class OllamaPsTest(unittest.TestCase):
    """``/api/ps``: saber qué hay cargado y no descargar lo que no lo está."""

    def test_fetch_loaded_devuelve_los_nombres(self) -> None:
        resp = mock.Mock()
        resp.json.return_value = {"models": [{"name": "qwen3.5:9b"}, {"model": "otro"}]}
        with mock.patch.object(ollama_api.requests, "get", return_value=resp) as get:
            self.assertEqual(ollama_api.fetch_loaded("http://x:1"), ["qwen3.5:9b", "otro"])
        self.assertEqual(get.call_args.args[0], "http://x:1/api/ps")

    def test_is_loaded_acepta_el_nombre_con_y_sin_tag(self) -> None:
        nombres = ["qwen3.5:9b"]
        with mock.patch.object(ollama_api, "fetch_loaded", return_value=nombres):
            self.assertTrue(ollama_api.is_loaded("http://x:1", "qwen3.5:9b"))
            self.assertTrue(ollama_api.is_loaded("http://x:1", "qwen3.5"))
            self.assertFalse(ollama_api.is_loaded("http://x:1", "qwen3.6:35b"))

    def test_is_loaded_falso_si_la_consulta_falla(self) -> None:
        with mock.patch.object(
            ollama_api,
            "fetch_loaded",
            side_effect=requests.RequestException("caido"),
        ):
            self.assertFalse(ollama_api.is_loaded("http://x:1", "qwen3.5:9b"))


class UnloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = OllamaEngine()

    def test_no_descarga_si_el_modelo_no_esta_cargado(self) -> None:
        # Este era el caso del log: "descargado de VRAM" de un modelo que nunca
        # se había usado (y que Ollama cargaba para poder descargarlo).
        with mock.patch.object(
            ollama_api, "is_loaded", return_value=False
        ), mock.patch.object(ollama_mod.requests, "post") as post:
            self.assertTrue(self.engine.unload_model())
        post.assert_not_called()

    def test_descarga_si_esta_cargado(self) -> None:
        with mock.patch.object(
            ollama_api, "is_loaded", return_value=True
        ), mock.patch.object(ollama_mod.requests, "post") as post:
            self.assertTrue(self.engine.unload_model())
        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], 0)
        self.assertEqual(
            post.call_args.kwargs["json"]["model"], self.engine._default_model
        )

    def test_usa_el_modelo_usado_si_lo_hay(self) -> None:
        payload = {"message": {"content": "hola"}}
        with mock.patch.object(
            ollama_mod.requests, "post", return_value=_response(payload)
        ):
            self.engine.chat([{"role": "user", "content": "hola"}])
        with mock.patch.object(
            ollama_api, "is_loaded", return_value=True
        ), mock.patch.object(ollama_mod.requests, "post") as post:
            self.engine.unload_model()
        self.assertEqual(post.call_args.kwargs["json"]["model"], self.engine._default_model)
        self.assertEqual(self.engine._last_used_model, self.engine._default_model)


if __name__ == "__main__":
    unittest.main()

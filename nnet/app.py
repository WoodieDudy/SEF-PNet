import io
import os
import numpy as np
import streamlit as st
import torch as th
import soundfile as sf
from scipy.signal import resample_poly
from audio_recorder_streamlit import audio_recorder

CHECKPOINT_DIR = "/home/gk/projects/sasung/exp_dpccn-8khz-v2-tcn-causal2"
ENCODER_PATH = "/home/gk/projects/sasung/voxceleb_resnet34_avg_model.pt"
FS = 8000
DEVICE_STR = "cpu"

from DPCCN_offline import DenseUNet as Model
from libs.utils import load_json

st.set_page_config(page_title="DPCCN Target Speaker Extraction", layout="wide")

def to_mono_float32(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / 32768.0
    else:
        y = y.astype(np.float32)
    return y

def ensure_sr(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    return resample_poly(y, sr_out, sr_in).astype(np.float32)

def align_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    if b.shape[0] < n:
        b = np.concatenate([b, np.zeros(n - b.shape[0], dtype=np.float32)], axis=0)
    elif b.shape[0] > n:
        b = b[:n]
    return a, b

def wav_bytes_from_array(y: np.ndarray, sr: int) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, y, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()

@st.cache_resource(show_spinner=True)
def load_model(checkpoint_dir: str, device_str: str):
    if not checkpoint_dir.endswith("/"):
        checkpoint_dir += "/"
    nnet_conf = load_json(checkpoint_dir, "mdl.json")
    nnet = Model(**nnet_conf, encoder_path=ENCODER_PATH)
    cpt_path = os.path.join(checkpoint_dir, "best.pt.tar")
    cpt = th.load(cpt_path, map_location="cpu")
    nnet.load_state_dict(cpt["model_state_dict"], strict=False)
    device = th.device(device_str) if device_str.startswith("cuda") else th.device("cpu")
    if str(device) != "cpu":
        nnet = nnet.to(device)
    nnet.eval()
    return nnet, device

def run_model(nnet: th.nn.Module, device: th.device, mix: np.ndarray, aux: np.ndarray) -> np.ndarray:
    nnet.reset()
    mix_t = th.from_numpy(mix).float().unsqueeze(0).to(device)
    aux_t = th.from_numpy(aux).float().unsqueeze(0).to(device)
    with th.no_grad():
        est = nnet(mix_t, aux_t)
    est_np = est.squeeze().detach().cpu().numpy().astype(np.float32)
    return est_np[: mix.shape[0]]

st.title("🎙️ Удаление посторонних голосов")

nnet, device = load_model(CHECKPOINT_DIR, DEVICE_STR)

tab_upload, tab_record = st.tabs(["⬆️ Загрузка файлов", "🎤 Запись с микрофона"])

with tab_upload:
    st.subheader("Загрузка файлов")
    c1, c2 = st.columns(2)
    with c1:
        mix_file = st.file_uploader("Аудио для очистки (WAV)", type=["wav"], key="mix_upl")
    with c2:
        aux_file = st.file_uploader("Пример голоса который нужно оставить (WAV)", type=["wav"], key="aux_upl")

    go_upload = st.button("▶️ Инференс (загруженные файлы)", type="primary", disabled=not (mix_file and aux_file))
    if go_upload:
        mix_y, mix_sr = sf.read(mix_file, dtype="float32")
        aux_y, aux_sr = sf.read(aux_file, dtype="float32")

        mix = ensure_sr(to_mono_float32(mix_y), mix_sr, FS)
        aux = ensure_sr(to_mono_float32(aux_y), aux_sr, FS)
        mix, aux = align_lengths(mix, aux)

        with st.spinner("Гоним через сеть…"):
            est = run_model(nnet, device, mix, aux)

        st.write("**Аудио для очистки (вход)**")
        st.audio(wav_bytes_from_array(mix, FS), format="audio/wav")
        st.write("**Пример голоса (вход)**")
        st.audio(wav_bytes_from_array(aux, FS), format="audio/wav")
        st.write("**Результат модели**")
        out_bytes = wav_bytes_from_array(est, FS)
        st.audio(out_bytes, format="audio/wav")
        st.download_button("💾 Скачать результат (WAV)", data=out_bytes, file_name="estimate.wav", mime="audio/wav")

with tab_record:
    st.subheader("Запись")
    st.caption("Нажми на кнопку, чтобы начать/остановить. Если первый клик дал 0 сек — просто запиши ещё раз (пустые записи игнорируются).")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Аудио для очистки**")
        mix_bytes = audio_recorder(
            text="Нажми для записи / остановки",
            recording_color="#ff4b4b",
            neutral_color="#6c757d",
            icon_size="2x",
            sample_rate=48000,
            key="rec_mix_simple",
        )
        if mix_bytes:
            if len(mix_bytes) < 2048:
                st.info("Похоже, запись пустая/слишком короткая. Нажми ещё раз.")
            else:
                y, sr = sf.read(io.BytesIO(mix_bytes), dtype="float32")
                y = ensure_sr(to_mono_float32(y), sr, FS)
                st.session_state["mix_rec"] = y
                st.audio(wav_bytes_from_array(y, FS), format="audio/wav")

    with c2:
        st.markdown("**Пример голоса который нужно оставить**")
        aux_bytes = audio_recorder(
            text="Нажми для записи / остановки",
            recording_color="#ff4b4b",
            neutral_color="#6c757d",
            icon_size="2x",
            sample_rate=48000,
            key="rec_aux_simple",
        )
        if aux_bytes:
            if len(aux_bytes) < 2048:
                st.info("Похоже, запись пустая/слишком короткая. Нажми ещё раз.")
            else:
                y, sr = sf.read(io.BytesIO(aux_bytes), dtype="float32")
                y = ensure_sr(to_mono_float32(y), sr, FS)
                st.session_state["aux_rec"] = y
                st.audio(wav_bytes_from_array(y, FS), format="audio/wav")

    go_rec = st.button(
        "▶️ Инференс (записанные фрагменты)",
        type="primary",
        disabled=not ("mix_rec" in st.session_state and "aux_rec" in st.session_state),
    )
    if go_rec:
        mix = st.session_state["mix_rec"]
        aux = st.session_state["aux_rec"]
        mix, aux = align_lengths(mix, aux)
        with st.spinner("Гоним через сеть…"):
            est = run_model(nnet, device, mix, aux)
        out_bytes = wav_bytes_from_array(est, FS)
        st.audio(out_bytes, format="audio/wav")
        st.download_button("💾 Скачать результат (WAV)", data=out_bytes, file_name="estimate.wav", mime="audio/wav")

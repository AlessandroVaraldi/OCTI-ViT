#!/usr/bin/env python3
# ============================================================
#  test_vit_webcam.py ― Live CIFAR-100 / ImageNet classifier
#  Adatta automaticamente la risoluzione in base all'ONNX.
# ============================================================

import argparse, cv2, numpy as np, onnxruntime as ort, time

# ------------------------------------------------------------
#  CIFAR-100 class names (official order)
# ------------------------------------------------------------
CIFAR100_CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard',
    'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

# ------------------------------------------------------------
#  Normalization constants (CIFAR-100; vanno bene anche a 224)
# ------------------------------------------------------------
MEAN = np.array([0.5071, 0.4866, 0.4409], dtype=np.float32)
STD  = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)

# ### NEW: preprocessing parametrico su H×W
def make_preprocess(H, W):
    """Crea una funzione di preprocess che produce un tensore 1×3×H×W float32 NCHW."""
    def _preprocess(frame_bgr):
        img = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD                      # HWC, float32
        img = img.transpose(2, 0, 1)[None, ...]       # 1×3×H×W
        return img
    return _preprocess

def main(args):
    # -------------------------
    #  ONNX Runtime session
    # -------------------------
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.cuda else ['CPUExecutionProvider']
    sess = ort.InferenceSession(args.model, providers=providers)

    # ### NEW: leggi nome input/output e SHAPE attesa
    inp_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    input_name  = inp_meta.name
    output_name = out_meta.name

    # inp_meta.shape di solito è [None, 3, H, W]; gestiamo eventuali None
    shape = inp_meta.shape
    # Converti dimensioni dinamiche (None o stringhe) in -1
    dims = [d if isinstance(d, int) else -1 for d in shape]
    _, C, H, W = dims if len(dims) == 4 else (1, 3, 224, 224)
    if H == -1 or W == -1:
        # Fallback ragionevole se l’ONNX ha dimensioni dinamiche non specificate
        H = W = 224

    print(f"Loaded ONNX model: {args.model}")
    print(f"Execution providers: {sess.get_providers()}")
    print(f"Model expects input: [N, {C}, {H}, {W}]")

    # ### NEW: scegli le label in base al numero di classi dell’output
    out_shape = out_meta.shape
    num_classes = out_shape[-1] if isinstance(out_shape[-1], int) else None
    if num_classes == 100:
        LABELS = CIFAR100_CLASSES
    else:
        LABELS = None
        if num_classes is not None:
            print(f"⚠️  Modello con {num_classes} classi: etichette non note, mostro solo l’indice.")
        else:
            print("⚠️  Output dinamico: etichette non note, mostro solo l’indice.")

    # Preprocess parametrico
    preprocess = make_preprocess(H, W)

    # -------------------------
    #  Webcam init
    # -------------------------
    cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if args.win else 0)
    if not cam.isOpened():
        raise RuntimeError("❌  Cannot open webcam")
    print("✅  Webcam opened. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("⚠️  Frame grab failed, retrying ...")
                continue

            # Pre-process and run inference
            inp = preprocess(frame)
            logits = sess.run([output_name], {input_name: inp})[0]  # (1, num_classes)
            pred  = int(np.argmax(logits, axis=1))
            conf  = float(np.max(logits, axis=1))  # opzionale: se è logit, non è softmax

            # Etichetta da mostrare
            if LABELS:
                label = LABELS[pred]
            else:
                label = f"class {pred}"

            # Overlay label (top-left corner) con piccola “ombra” per leggibilità
            text = f"{label}"
            cv2.putText(frame, text, (11, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("CIFAR-100 / ImageNet Live Classification", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("👋  Bye!")

# ------------------------------------------------------------
#  CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Live classifier using an ONNX ViT")
    ap.add_argument("--model",  default="vit_cifar100.onnx", help="Path to ONNX model")
    ap.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    ap.add_argument("--cuda",   action="store_true", help="Use CUDAExecutionProvider if available")
    ap.add_argument("--win",    action="store_true", help="Use DirectShow backend on Windows")
    args = ap.parse_args()
    main(args)

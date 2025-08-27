#!/usr/bin/env python3
# ============================================================
#  test_vit_webcam.py ― Live CIFAR-100 classification
# ------------------------------------------------------------
#  • Loads vit_cifar100.onnx exported by train_tiny_vit.py
#  • Captures frames from the default webcam (Device 0)
#  • Resizes to 32×32, normalizes with CIFAR-100 stats,
#    runs inference via onnxruntime, and overlays the top-1 label
#  • Press  q  to quit
# ============================================================

import argparse, cv2, numpy as np, onnxruntime as ort, time

# ------------------------------------------------------------
#  CIFAR-100 class names (official order)
#  Hard-coded to avoid extra dependencies at runtime.
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
#  Normalization constants used during training
# ------------------------------------------------------------
MEAN = np.array([0.5071, 0.4866, 0.4409], dtype=np.float32)
STD  = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)

def preprocess(frame_bgr):
    """Resize to 32×32, convert to RGB, normalize, create NCHW tensor."""
    img = cv2.resize(frame_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD                      # HWC, float32
    img = img.transpose(2, 0, 1)[None, ...]       # 1×3×32×32
    return img

def main(args):
    # -------------------------
    #  ONNX Runtime session
    # -------------------------
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.cuda else ['CPUExecutionProvider']
    sess = ort.InferenceSession(args.model, providers=providers)
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"Loaded ONNX model: {args.model}")
    print(f"Execution providers: {sess.get_providers()}")

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
            logits = sess.run([output_name], {input_name: inp})[0]
            pred  = int(np.argmax(logits))
            label = CIFAR100_CLASSES[pred]

            # Overlay label (top-left corner)
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                        cv2.LINE_AA)

            cv2.imshow("CIFAR-100 Live Classification", frame)

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
    ap = argparse.ArgumentParser(description="Live CIFAR-100 classifier using vit_cifar100.onnx")
    ap.add_argument("--model",  default="vit_cifar100.onnx", help="Path to ONNX model")
    ap.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    ap.add_argument("--cuda",   action="store_true", help="Use CUDAExecutionProvider if available")
    ap.add_argument("--win",    action="store_true", help="Use DirectShow backend on Windows")
    args = ap.parse_args()
    main(args)

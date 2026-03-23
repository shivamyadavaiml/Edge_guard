import argparse
import os
import json
import hashlib
import sys
import traceback

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def human_size(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"

def inspect_torch_file(path):
    try:
        import torch
    except Exception:
        print("PyTorch not installed. Install torch to inspect .pt files deeply.")
        return {"error":"torch-missing"}

    try:
        data = torch.load(path, map_location='cpu')
    except Exception as e:
        return {"error":"load-failed","exception":repr(e)}

    info = {"type": type(data).__name__}

    if isinstance(data, dict):
        info['keys'] = list(data.keys())

        # state_dict style
        if 'state_dict' in data and isinstance(data['state_dict'], dict):
            sd = data['state_dict']
            info['state_dict_keys'] = len(sd)
            total_params = 0
            top = []
            for i,(k,v) in enumerate(sd.items()):
                try:
                    shape = list(v.size())
                    num = v.numel()
                except Exception:
                    shape = str(type(v))
                    num = 0
                total_params += num
                if i < 20:
                    top.append({'name':k,'shape':shape,'numel':num})
            info['total_parameters'] = total_params
            info['top_keys'] = top

        # direct state_dict saved as top-level (common in some exports)
        elif any(k.endswith('.weight') or k.endswith('.bias') for k in data.keys()):
            sd_keys = [k for k in data.keys()]
            info['state_dict_like'] = True
            info['state_dict_key_count'] = len(sd_keys)

        # yaml/meta info
        for meta_key in ('yaml','model', 'names', 'nc', 'epoch', 'optimizer'):
            if meta_key in data:
                try:
                    info[f'meta_{meta_key}'] = data[meta_key]
                except Exception:
                    info[f'meta_{meta_key}'] = type(data[meta_key]).__name__

    else:
        info['repr'] = repr(data)[:1000]

    return info

def try_ultralytics(path):
    res = {}
    try:
        from ultralytics import YOLO
        y = YOLO(path)
        res['ultralytics_loaded'] = True
        # general attrs
        for attr in ('model','names','nc','task'):
            if hasattr(y, attr):
                try:
                    res[attr] = getattr(y, attr)
                except Exception:
                    res[attr] = type(getattr(y, attr)).__name__

        # model internals
        try:
            m = getattr(y, 'model', None)
            if m is not None:
                # number of parameters
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                res['model_parameters'] = {'total': total, 'trainable': trainable}
                # stride and anchors if present
                for a in ('stride','anchors'):
                    if hasattr(m, a):
                        try:
                            res[a] = getattr(m, a)
                        except Exception:
                            res[a] = str(type(getattr(m, a)))
        except Exception:
            res['model_inspect_error'] = traceback.format_exc()

    except Exception as e:
        res['ultralytics_error'] = repr(e)
    return res

def build_report(path, out_json=None):
    report = {}
    report['path'] = path
    try:
        st = os.stat(path)
        report['size_bytes'] = st.st_size
        report['size_human'] = human_size(st.st_size)
        report['sha256'] = sha256_of_file(path)
    except Exception as e:
        report['file_error'] = repr(e)
        return report

    # Torch-based inspection
    report['torch_inspect'] = inspect_torch_file(path)

    # Try ultralytics if available for richer info
    report['ultralytics_inspect'] = try_ultralytics(path)

    if out_json:
        try:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            report['json_write_error'] = repr(e)

    return report

def main():
    p = argparse.ArgumentParser(description='Inspect a YOLO / PyTorch model file and print metadata.')
    p.add_argument('model', nargs='?', default='yolov8n.pt', help='Path to model file (default: yolov8n.pt)')
    p.add_argument('--out', '-o', help='Write JSON report to file')
    args = p.parse_args()

    path = args.model
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        sys.exit(2)

    report = build_report(path, args.out)

    # Print concise human-friendly summary
    print("--- Model File Summary ---")
    print(f"Path: {report.get('path')}")
    if 'size_human' in report:
        print(f"Size: {report['size_human']} ({report.get('size_bytes')} bytes)")
    if 'sha256' in report:
        print(f"SHA256: {report['sha256']}")

    ti = report.get('torch_inspect', {})
    if ti.get('error'):
        print("Torch inspection error:", ti.get('exception', ti.get('error')))
    else:
        print('Torch object type:', ti.get('type'))
        if 'keys' in ti:
            print('Top-level keys:', ', '.join(ti['keys']))
        if 'total_parameters' in ti:
            print('Parameters (from state_dict):', ti['total_parameters'])

    ui = report.get('ultralytics_inspect', {})
    if ui.get('ultralytics_loaded'):
        mp = ui.get('model_parameters')
        if mp:
            print(f"Parameters (ultralytics): total={mp['total']}, trainable={mp['trainable']}")
        if 'names' in ui:
            try:
                # names may be dict or list
                n = ui['names']
                if isinstance(n, dict):
                    names_list = [n[k] for k in sorted(n.keys(), key=lambda x:int(x))]
                else:
                    names_list = list(n)
                print('Class names:', len(names_list), 'classes')
            except Exception:
                print('Class names: (could not parse)')
    else:
        if 'ultralytics_error' in ui:
            print('Ultralytics not available or failed:', ui['ultralytics_error'])

    if args.out:
        print(f"Full JSON report written to: {args.out}")

    # Optionally print detailed JSON to stdout when not writing to file
    print('\n--- Detailed JSON Summary ---')
    print(json.dumps(report, indent=2, default=str)[:20000])

if __name__ == '__main__':
    main()

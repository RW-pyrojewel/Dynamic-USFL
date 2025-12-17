import csv
import os
import shutil
import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
TARGET_FIELDS = {"comp_time_client", "comp_client_total", "comp_cost_client"}

if not os.path.isdir(ROOT):
    print(f"Logs folder not found: {ROOT}")
    raise SystemExit(1)

files = glob.glob(os.path.join(ROOT, '**', '*.csv'), recursive=True)
print(f"Found {len(files)} CSV files under {ROOT}")

for path in files:
    try:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            fields = [fn for fn in fieldnames if fn in TARGET_FIELDS]
            if not fields:
                continue
            rows = list(reader)
    except Exception as e:
        print(f"Skip (read error) {path}: {e}")
        continue

    # backup
    bak = path + '.bak'
    if not os.path.exists(bak):
        try:
            shutil.copy2(path, bak)
        except Exception as e:
            print(f"Backup failed for {path}: {e}")
            continue

    changed = False
    for r in rows:
        for fld in fields:
            raw = r.get(fld, '')
            if raw is None:
                continue
            s = raw.strip()
            if s == '':
                continue
            # remove thousands separators
            s2 = s.replace(',', '')
            try:
                v = float(s2)
            except Exception:
                # try to handle scientific with stray chars
                try:
                    v = float(''.join(ch for ch in s2 if ch in '0123456789+-.eE'))
                except Exception:
                    continue
            nv = v * 10.0
            # preserve integer-like formatting
            if abs(nv - round(nv)) < 1e-9:
                r[fld] = str(int(round(nv)))
            else:
                # compact representation
                r[fld] = '{:.6g}'.format(nv)
            changed = True

    if not changed:
        continue

    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        shutil.move(tmp, path)
        print(f"Updated {path} (backup at {bak})")
    except Exception as e:
        print(f"Failed to write {path}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        # try to restore backup
        try:
            if os.path.exists(bak):
                shutil.copy2(bak, path)
        except Exception:
            pass

print("Done.")

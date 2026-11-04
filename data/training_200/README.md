# Experimental ₹200 classifier data

- `genuine/` contains the 12 photos supplied as genuine references.
- `fake/` contains the 11 photos the user labeled as fake.

The labels have not been independently authenticated. Some images depict
specimen/sample graphics, and the photos may not represent independent physical
notes. The classifier's training-set fit is not a held-out test result; do not
interpret its scores as calibrated probabilities or proof of authenticity.

Retrain with:

```powershell
.\venv\Scripts\python.exe train_200_authenticity.py
```

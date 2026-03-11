# Neutral Atoms Simulator

Interactive visualization for neutral atom array reconfiguration and gate execution planning.

## Architecture

```
project-root/
├── neutral_atoms/           # Your Python RL framework
│   ├── __init__.py
│   ├── moves.py             # parallel_groups, count_groups, canonicalize_moves
│   ├── types.py
│   └── ...
└── atom-viz/                # This visualization tool
    ├── api.py               # FastAPI server (imports from neutral_atoms)
    ├── generate_template.py
    ├── src/                 # React frontend
    └── ...
```

## Quick Start

### Prerequisites

**Node.js 18+** and **Python 3.10+** with:
```bash
pip install fastapi uvicorn torch
npm install
```

### Running

**Terminal 1: Python API**
```bash
cd atom-viz
python api.py
# Runs on http://localhost:8000
```

**Terminal 2: React Frontend**
```bash
cd atom-viz
npm run dev
# Runs on http://localhost:5173
```

### Remote Server

```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 user@your-server
```

## Features

### Interactive Editing
- **Click atoms** in the current layer's gates to select
- **Click empty cells** to draw reconfiguration moves
- **Click arrows** to delete moves
- **← → keys** to navigate layers

### Visual Feedback
- **Ghost atoms**: Moved atoms appear as dashed outlines at destination
- **Parallel groups**: Same color = can execute together
- **Inactive atoms**: Grey (not in current gates, cannot be moved)
- **Real-time costs**: Updates as you edit

### Task Editing
- Click **Edit** next to gates display
- Enter JSON format: `[[0,1], [2,3]]`
- Press Enter or click Save

### Validation
- Cannot move atoms not in current gates
- Cannot move two atoms to same destination
- Cannot move to occupied cell (unless that atom is also moving)
- Later layers auto-validate when earlier layers change

## File Structure

```
atom-viz/
├── api.py                  # FastAPI (imports neutral_atoms.moves)
├── generate_template.py    # Create random data.json
├── public/
│   └── data.json          # Loaded on startup
├── src/
│   ├── App.tsx            # Main app
│   ├── Grid.tsx           # SVG grid with ghost atoms
│   ├── api.ts             # API client
│   └── types.ts           # TypeScript types
└── requirements.txt
```

## JSON Format

```json
{
  "board": {
    "rows": 3,
    "cols": 10,
    "initialAtoms": {
      "0": {"row": 0, "col": 1},
      "1": {"row": 1, "col": 2}
    }
  },
  "circuit": [
    [[0, 8], [6, 2]],
    [[0, 2], [1, 5]]
  ],
  "plan": [
    [],
    [{"atom": 2, "from": {"row": 0, "col": 3}, "to": {"row": 1, "col": 3}}]
  ]
}
```

## Integrating with neutral_atoms

The API imports directly from your `neutral_atoms` package:

```python
from neutral_atoms.moves import parallel_groups, count_groups
```

- **Reconfig moves**: `canonicalize=False`
- **Gate moves**: `canonicalize=True` (direction-flexible)

If `neutral_atoms` is not found, a fallback implementation is used (without canonicalization).

# Plotting Functions

Derive provides plotting functions built on matplotlib.

## Plot - 2D Function Plots

Plot functions over a range.

```python
from derive import *

x = Symbol('x')

# Basic plot
Plot(Sin(x), (x, 0, 2*Pi))
```

### Plot Options

```python
# Custom styling
Plot(Sin(x), (x, 0, 2*Pi),
     PlotStyle='red',
     PlotLabel='Sine Wave',
     AxesLabel=['x', 'y'],
     GridLines=True)
```

### Multiple Functions

```python
# Plot multiple functions
Plot([Sin(x), Cos(x)], (x, 0, 2*Pi),
     PlotLabel='Trig Functions')
```

## ListPlot - Data Visualization

Plot discrete data points.

```python
# Simple data
data = [[1, 1], [2, 4], [3, 9], [4, 16]]
ListPlot(data)

# Y-values only (x starts at 1)
ListPlot([1, 4, 9, 16, 25])
```

### ListLinePlot

Connect points with lines.

```python
ListLinePlot([[1, 1], [2, 4], [3, 9], [4, 16]])
```

## ParametricPlot - Parametric Curves

Plot curves defined parametrically.

```python
t = Symbol('t')

# Circle
ParametricPlot([Cos(t), Sin(t)], (t, 0, 2*Pi))

# Lissajous figure
ParametricPlot([Sin(3*t), Cos(2*t)], (t, 0, 2*Pi),
               PlotLabel='Lissajous')
```

## Show - Combining Plots

```python
# Create individual plots
p1 = Plot(Sin(x), (x, 0, 2*Pi), PlotStyle='blue')
p2 = Plot(Cos(x), (x, 0, 2*Pi), PlotStyle='red')

# Combine them
Show(p1, p2)
```

## Plot Options Reference

| Option | Description | Example |
|--------|-------------|---------|
| `PlotStyle` | Line color/style | `'red'`, `'b--'`, `{'color': 'blue', 'linewidth': 2}` |
| `PlotLabel` | Plot title | `'My Plot'` |
| `AxesLabel` | Axis labels | `['x', 'y']` |
| `PlotRange` | Axis limits | `'All'`, `[[-1, 1], [-2, 2]]` |
| `GridLines` | Show grid | `True`, `False` |
| `AspectRatio` | Height/width ratio | `'Automatic'`, `1` |
| `Filling` | Fill area | `True`, `'Axis'` |

## Saving Plots

Use `Export` to save plots to files.

```python
fig = Plot(Sin(x), (x, 0, 2*Pi))
Export('sin_plot.pdf', fig)
Export('sin_plot.png', fig)
```

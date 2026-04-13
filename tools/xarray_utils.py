from typing import Union
import xarray as xr
from pathlib import Path
import html as html_module

xr.set_options(
    display_width=120,          # wider output
    display_max_rows=50,        # more rows before truncation
    display_expand_data_vars=True,
    display_expand_coords=True,
    display_expand_attrs=True,
)

def repr_html_with_sizes(ds):
    lines = []
    for name, var in ds.variables.items():
        size_mb = var.nbytes / 1e6
        lines.append(f"<tr><td><b>{html_module.escape(name)}</b></td>"
                      f"<td>{html_module.escape(str(var.dims))}</td>"
                      f"<td>{var.dtype}</td>"
                      f"<td>{size_mb:.1f} MB</td></tr>")
    
    table = f"<table>{''.join(lines)}</table>"
    total_gb = ds.nbytes / 1e9
    
    return (f"<h4>xarray.Dataset — Size: {total_gb:.1f} GB</h4>"
            + ds._repr_html_()
            + "<h4>Variable Sizes</h4>"
            + table)


def print_and_save_data_overview(da: Union[xr.DataArray, xr.Dataset], output_file: str):
    output_file = Path(output_file).expanduser()
    if output_file.suffix == '' or output_file.suffix != '.html':
        output_file = output_file.with_suffix('.html')
        
    with open(output_file, "w") as f:
        f.write(repr_html_with_sizes(da))
    print(f"Saved to {output_file}")
    
    
def make_overviews(h5_file: str):
    h5_file = Path(h5_file).expanduser()
    ds = xr.open_dataset(h5_file, engine='h5netcdf', chunks={})
    print_and_save_data_overview(ds, h5_file.with_suffix('.html'))
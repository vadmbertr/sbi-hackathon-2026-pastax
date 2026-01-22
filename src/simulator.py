import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxtyping import Array, Float, Real
import numpy as np
from pastax.gridded import Gridded
from pastax.simulator import DeterministicSimulator
from pastax.trajectory import Location
from pastax.utils import meters_to_degrees
import xarray as xr


def sanitize(arr):
    return jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def dynamical_model(
    t: Real[Array, ""], y: Float[Array, "2"], args: (Gridded, Gridded), params: Float[Array, "3"]
) -> Float[Array, "2"]:
    def interp(field, variables, t, lat, lon):
        uv_dict = field.interp(*variables, time=t, latitude=lat, longitude=lon)
        return jnp.asarray([uv_dict[k] for k in variables])
    
    def geostrophy(vu):
            return sanitize(vu)
        
    def ekman(tytx, latitude):
        txy = tytx[-1] + 1j * tytx[0]
        alpha = beta_e * jnp.exp(-1j * jnp.deg2rad(theta_e) * jnp.sign(latitude))

        uv_e = txy * alpha
        vu = jnp.array([uv_e.imag, uv_e.real])

        return sanitize(vu)

    def leeway(vu):
        return beta_w * sanitize(vu)
    
    latitude, longitude = y
    gridded_duacs, gridded_era5 = args
    beta_e, theta_e, beta_w = params    

    duacs_vu = interp(gridded_duacs, ("v", "u"), t, latitude, longitude)
    era5_tytx_vu = interp(gridded_era5, ("ty", "tx", "v", "u"), t, latitude, longitude)

    dlatlon = geostrophy(duacs_vu) + ekman(era5_tytx_vu[:2], latitude) + leeway(era5_tytx_vu[2:])
    
    if gridded_duacs.is_spherical_mesh and not gridded_duacs.use_degrees:
        dlatlon = meters_to_degrees(dlatlon, latitude=latitude)

    return dlatlon


def simulate_trajectories(
    duacs_ds: xr.Dataset,
    era5_ds: xr.Dataset,
    x0: Float[Array, "2"],  # lon, lat
    ts: Float[Array, "T"],  # timestamps
    sampled_parameters: Float[Array, "N 3"]
):
    simulator = DeterministicSimulator()
    dt0 = 15 * 60  # 15 minutes in seconds

    gridded_duacs = Gridded.from_xarray(
        duacs_ds, 
        fields={"u": "ugos", "v": "vgos"},
        coordinates={"time": "time", "latitude": "latitude", "longitude": "longitude"}
    )
    gridded_era5 = Gridded.from_xarray(
        era5_ds, 
        fields={
            "tx": "eastward_stress", "ty": "northward_stress",
            "u": "eastward_wind", "v": "northward_wind"
        },
        coordinates={"time": "time", "latitude": "latitude", "longitude": "longitude"}
    )

    x0 = Location(jnp.asarray(x0)[::-1])  # convert to (lat, lon)
    ts = jnp.asarray(ts.astype("datetime64[s]").astype(np.float64))
    sampled_parameters = jnp.asarray(sampled_parameters)

    def simulate_single_params(params: Float[Array, "3"]):
        dynamics = lambda t, y, args: dynamical_model(t, y, args, params)

        trajectory = simulator(dynamics=dynamics, args=(gridded_duacs, gridded_era5), x0=x0, ts=ts, dt0=dt0)
        latlon = trajectory.locations.value

        return latlon

    trajectories = jax.vmap(simulate_single_params, in_axes=0)(sampled_parameters)  # shape (N, T, 2)

    return trajectories[:, :, ::-1]  # return as (lon, lat)

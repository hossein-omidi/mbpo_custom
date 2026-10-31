"""One CLI contract for both workspaces, without changing historical defaults.

Choose --power-model pvwatts_sapm explicitly for new experiments. Identical
flags must be passed to training and evaluation. Defaults for wind/mount/gamma
are provisional and must be described in scientific reporting.
"""
from __future__ import annotations


def add_power_cli_arguments(parser):
    group = parser.add_argument_group('PV physical model (applies to train and evaluation)')
    group.add_argument('--power-model', choices=('legacy', 'pvwatts_sapm'), default='pvwatts_sapm',
                       help='Legacy remains default for reproducibility. Use pvwatts_sapm for new experiments.')
    group.add_argument('--wind-reference-height-m', type=float, default=2.0,
                       help='PROVISIONAL NSRDB measurement height; confirm source dataset')
    group.add_argument('--gamma-pdc', type=float, default=-0.004,
                       help='PROVISIONAL module DC power-temperature coefficient [1/degree C]')
    group.add_argument('--sapm-mount', default='open_rack_glass_polymer',
                       choices=('open_rack_glass_glass','close_mount_glass_glass',
                                'open_rack_glass_polymer','insulated_back_glass_polymer'),
                       help='SAPM steady-state temperature mounting preset')
    group.add_argument('--wind-powerlaw-exponent',type=float,default=0.14,
                       help='PROVISIONAL wind power-law exponent for conversion to 10 m')
    return parser


def power_cli_kwargs(args):
    """Return arguments for SAC_workspace config and MBPO_workspace config."""
    return dict(power_model=args.power_model,
                wind_reference_height_m=args.wind_reference_height_m,
                gamma_pdc=args.gamma_pdc,
                sapm_mount=args.sapm_mount,
                wind_powerlaw_exponent=args.wind_powerlaw_exponent)

#--------------------------------
# plot
#--------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
import matplotlib.path as mpath # added by CN

#--------------------------------
# basemap
#--------------------------------

def plot_basemap(
        region='Global',
        **kwargs,
        ):
    """Plot basemap

    :region:    (str) region name
    :**kwargs:  keyword arguments
    :return:    (cartopy.mpl.geoaxes.GeoAxes)

    """
    # plot basemap
    switcher = {
            'Global': _plot_basemap_global,
            'SouthernOcean': _plot_basemap_southern_ocean,
            'SO_highLats': _plot_basemap_high_lat_southern_ocean,
            'SO_highLats40': _plot_basemap_high_lat_southern_ocean40,
            'SO_highLats45': _plot_basemap_high_lat_southern_ocean45,
            'SouthernOcean_v2': _plot_basemap_southern_ocean_v2,
            'DrakePassage': _plot_basemap_drake_passage,
            'LabSea': _plot_basemap_labsea,
            'NorthPacific': _plot_basemap_northpacific,
            'TropicalPacific': _plot_basemap_tropicalpacific,
            'TropicalPacificSmall': _plot_basemap_tropicalpacific_small,
            'TropicalAtlantic': _plot_basemap_tropicalatlantic,
            'east_pacific': _plot_basemap_eastpacific,
            'south_atlantic': _plot_basemap_southatlantic,
            'north_pacific': _plot_basemap_north_pacific,
            'Custom': _plot_basemap_region,
            }
    if region in switcher.keys():
        return switcher.get(region)(**kwargs)
    else:
        raise ValueError('Region \'{:s}\' not found.\n'.format(region) \
                + '- Supported region names:\n' \
                + '  ' + ', '.join(switcher.keys()))

def _plot_basemap_global(axis=None):
    """Plot basemap for global region

    """
    # global map
    if axis is None:
        m = plt.axes(projection=ccrs.PlateCarree(central_longitude=210.0)) #central_longitude=200.0
    else:
        m = axis
        m.projection = ccrs.PlateCarree(central_longitude=210.0) #central_longitude=200.0
    m.set_global()
    # plot land and coastlines, draw label meridians and parallels.
    m.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='gray')
    gl = m.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False
    return m

def _plot_basemap_southern_ocean(axis=None):
    """Plot basemap for Southern Ocean

    """
    if axis is None:
        print('axis not defined')
        m = plt.axes(projection=ccrs.SouthPolarStereo())
    else:
        print('axis defined')
        m = axis
        m.projection=ccrs.SouthPolarStereo()
    # Limit the map to 40 degrees latitude and above.
    m.set_extent([-180, 180, -90, -40], ccrs.PlateCarree())
    #m.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='gray')
    m.add_feature(cfeature.GSHHSFeature('intermediate'), zorder=1, edgecolor='black', facecolor='gray')
    m.plot([180, 180], [-90, -78.6],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    m.plot([0, 0], [-90, -70.5],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
 #   gl = m.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
 #   gl.xlocator = mticker.FixedLocator(np.linspace(-180, 180, 13))
 #   gl.right_labels = False
    return m

def _plot_basemap_high_lat_southern_ocean(axis=None):
    """Plot basemap for high-latitude Southern Ocean

    """
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
    if axis is None:
        print('axis not defined')
        m = plt.axes(projection=ccrs.SouthPolarStereo())
    else:
        print('axis defined')
        m = axis
        m.projection=ccrs.SouthPolarStereo()
    # Limit the map to 40 degrees latitude and above.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    m.set_boundary(circle, transform=m.transAxes)
    
    m.set_extent([-180, 180, -90, -55], ccrs.PlateCarree())
    #m.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='gray')
    m.add_feature(cfeature.GSHHSFeature('intermediate'), zorder=1, edgecolor='black', facecolor='gray')
    m.plot([180, 180], [-90, -78.6],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    m.plot([0, 0], [-90, -70.5],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    
 #   gl = m.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
 #   gl.xlocator = mticker.FixedLocator(np.linspace(-180, 180, 13))
 #   gl.right_labels = False
    gl=m.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.5,\
                    color='grey', alpha=0.8,linestyle='--')#,xlabels_bottom=True)
    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels=True
    #gl.xlines = True
    gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120]) 
    gl.ylocator = mticker.FixedLocator([-70,-65,-60])
    #gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.rotate_labels = False
    
    return m

def _plot_basemap_high_lat_southern_ocean45(axis=None):
    """Plot basemap for high-latitude Southern Ocean

    """
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
    if axis is None:
        print('axis not defined')
        m = plt.axes(projection=ccrs.SouthPolarStereo())
    else:
        print('axis defined')
        m = axis
        m.projection=ccrs.SouthPolarStereo()
    # Limit the map to 40 degrees latitude and above.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    m.set_boundary(circle, transform=m.transAxes)
    
    m.set_extent([-180, 180, -90, -45], ccrs.PlateCarree())
    #m.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='gray')
    m.add_feature(cfeature.GSHHSFeature('intermediate'), zorder=1, edgecolor='black', facecolor='gray')
    m.plot([180, 180], [-90, -78.6],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    m.plot([0, 0], [-90, -70.5],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    
 #   gl = m.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
 #   gl.xlocator = mticker.FixedLocator(np.linspace(-180, 180, 13))
 #   gl.right_labels = False
    gl=m.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.5,\
                    color='grey', alpha=0.8,linestyle='--')#,xlabels_bottom=True)
    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels=True
    #gl.xlines = True
    gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120]) 
    gl.ylocator = mticker.FixedLocator([-70,-60,-50])
    #gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.rotate_labels = False
    
    return m

def _plot_basemap_high_lat_southern_ocean40(axis=None):
    """Plot basemap for high-latitude Southern Ocean

    """
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
    if axis is None:
        #print('axis not defined')
        m = plt.axes(projection=ccrs.SouthPolarStereo())
    else:
        #print('axis defined')
        m = axis
        m.projection=ccrs.SouthPolarStereo()
    # Limit the map to 40 degrees latitude and above.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    m.set_boundary(circle, transform=m.transAxes)
    
    m.set_extent([-180, 180, -90, -40], ccrs.PlateCarree())
    #m.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='gray')
    m.add_feature(cfeature.GSHHSFeature('intermediate'), zorder=1, edgecolor='black', facecolor='gray')
    m.plot([180, 180], [-90, -78.6],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    m.plot([0, 0], [-90, -70.5],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    
 #   gl = m.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
 #   gl.xlocator = mticker.FixedLocator(np.linspace(-180, 180, 13))
 #   gl.right_labels = False
    gl=m.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.5,\
                    color='grey', alpha=0.8,linestyle='--')#,xlabels_bottom=True)
    gl.top_labels = True
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels=True
    #gl.xlines = True
    gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120]) 
    gl.ylocator = mticker.FixedLocator([-70,-60,-50])
    #gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.rotate_labels = False
    
    return m

def _plot_basemap_southern_ocean_v2(axis=None):
    """Plot basemap for Southern Ocean, v2 (modifed by CN)

    """
    if axis is None:
        m = plt.axes(projection=ccrs.SouthPolarStereo())
    else:
        m = axis
        m.projection=ccrs.SouthPolarStereo()
    # Limit the map to 40 degrees latitude and above.
    axis.set_extent([-180, 180, -90, -40], ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='gray')
    #m.set_extent([-180, 180, -90, -40], ccrs.PlateCarree())
    #m.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='gray')
    # added by CN:
   # theta = np.linspace(0, 2*np.pi, 100)
   # center, radius = [0.5, 0.5], 0.5
   # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
   # circle = mpath.Path(verts * radius + center)
   # plt.axis.set_boundary(circle, transform=plt.axis.transAxes) 
   # plt.axis.outline_patch.set_edgecolor('black') # black, white, None...
    #
    gl = m.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    #gl = m.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(np.linspace(-180, 180, 13))
    gl.right_labels = False
    return m

def _plot_basemap_drake_passage(axis=None):
    """Plot basemap for Drake Passage (modifed by CN)

    """ 
    return _plot_basemap_region(axis=axis, lon_min=270, lat_min=-75, \
                                lon_max=310, lat_max=-50, projection=ccrs.PlateCarree(central_longitude=290), \
                                xlocator=[-80,-60], \
                                ylocator=[-75,-65,-55])

def _plot_basemap_region(
        lon_min,
        lat_min,
        lon_max,
        lat_max,
        projection,
        axis=None,
        xlocator=None,
        ylocator=None,
        ):
    """Plot basemap for a particular region

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :lon_min:       (float) minimum longitude in degrees
    :lat_min:       (float) minimum latitude in degrees
    :lon_max:       (float) maximum longitude in degrees
    :lat_max:       (float) maximum latitude in degrees
    :projection:    (cartopy.crs.Projection) projection type
    :xlocator:      (array-like) x tick locations
    :ylocator:      (array-like) y tick locations
    :return:        (cartopy.mpl.geoaxes.GeoAxes)

    """
    # regional map
    if axis is None:
        print('axis not defined')
        m = plt.axes(projection=projection)
    else:
        print('axis defined')
        m = axis
        m.projection = projection
    m.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    #m.add_feature(cfeature.LAND, zorder=1, edgecolor='black', facecolor='gray')
    m.add_feature(cfeature.GSHHSFeature('intermediate'), zorder=1, edgecolor='black', facecolor='gray')
    m.plot([180, 180], [-90, -78.6],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    m.plot([0, 0], [-90, -70.5],
                 zorder=3, color='gray',
                 linewidth=2,
                 transform=ccrs.PlateCarree())
    gl = m.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    #gl = m.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha = 0.3)
    if xlocator is not None:
        gl.xlocator = mticker.FixedLocator(xlocator)
    if ylocator is not None:
        gl.ylocator = mticker.FixedLocator(ylocator)
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    return m

def _plot_basemap_labsea(axis=None):
    """Plot basemap for Labrador Sea

    """
    m = _plot_basemap_region(axis=axis, lon_min=296.0, lat_min=40.0, \
                             lon_max=336.0, lat_max=70.0, \
                             projection=ccrs.Orthographic(central_longitude=316.0, central_latitude=55), \
                             xlocator=np.linspace(-80,-30,6), \
                             ylocator=np.linspace(40,65,6))
    # workaround to turn off all the right and top labels
    gl = m.gridlines(draw_labels=False)
    gl.xlocator = mticker.FixedLocator([-20, -10])
    gl.ylocator = mticker.FixedLocator([70])
    return m

def _plot_basemap_northpacific(axis=None):
    """Plot basemap for North Pacific

    """
    return _plot_basemap_region(axis=axis, lon_min=130.0, lat_min=20.0, \
                                lon_max=195.0, lat_max=60.0, projection=ccrs.PlateCarree(central_longitude=165), \
                                xlocator=np.concatenate([np.linspace(130,180,6), np.linspace(-190,-200,2)]), \
                                ylocator=[20, 30, 40, 50, 60])

def _plot_basemap_tropicalpacific(axis=None):
    """Plot basemap for Tropical Pacific

    """
    return _plot_basemap_region(axis=axis, lon_min=130.0, lat_min=-25.0, \
                                lon_max=290.0, lat_max=25.0, projection=ccrs.PlateCarree(central_longitude=210), \
                                xlocator=[-160,-140,-120,-100,-80,140,160,180], \
                                ylocator=[-20, -10, 0, 10, 20])
#np.concatenate([np.linspace(130,180,6), np.linspace(-190,-70,13)])

def _plot_basemap_tropicalpacific_small(axis=None):
    """Plot basemap for Tropical Pacific (small)

    """
    return _plot_basemap_region(axis=axis, lon_min=159.9, lat_min=-10.1, \
                                lon_max=280.1, lat_max=10.1, projection=ccrs.PlateCarree(central_longitude=220), \
                                xlocator=np.concatenate([np.linspace(160,180,3), np.linspace(-190,-80,12)]), \
                                ylocator=[-10, 0, 10])

def _plot_basemap_tropicalatlantic(axis=None):
    """Plot basemap for Tropical Atlantic

    """
    return _plot_basemap_region(axis=axis, lon_min=310.0, lat_min=-20.0, \
                                lon_max=380.0, lat_max=20.0, projection=ccrs.PlateCarree(central_longitude=345), \
                                xlocator=np.concatenate([np.linspace(-50,0,6), np.linspace(10,20,2)]), \
                                ylocator=[-20, -10, 0, 10, 20])

def _plot_basemap_eastpacific(axis=None):
    """Plot basemap for Tropical Pacific, CN

    """
    return _plot_basemap_region(axis=axis, lon_min=230.0, lat_min=-42.0, \
                                lon_max=300.0, lat_max=2.0, projection=ccrs.PlateCarree(central_longitude=265), \
                                xlocator=np.concatenate([np.linspace(160,180,3), np.linspace(-190,-80,12)]), \
                                ylocator=[-40,-30,-20, -10, 0])

def _plot_basemap_southatlantic(axis=None):
    """Plot basemap for South Atlantic, CN

    """
    return _plot_basemap_region(axis=axis, lon_min=300.0, lat_min=-35.0, \
                                lon_max=360.0, lat_max=2.0, projection=ccrs.PlateCarree(central_longitude=330), \
                                xlocator=np.concatenate([np.linspace(160,180,3), np.linspace(-190,-80,12)]), \
                                ylocator=[-30,-20, -10, 0])

def _plot_basemap_north_pacific(axis=None):
    """Plot basemap for North Pacific, CN

    """
    return _plot_basemap_region(axis=axis, lon_min=189.0, lat_min=14.0, \
                                lon_max=241.0, lat_max=31.0, projection=ccrs.PlateCarree(central_longitude=210), \
                                xlocator=np.concatenate([np.linspace(160,180,3), np.linspace(-190,-80,12)]), \
                                ylocator=[15,30])
#lon_min=175.0, lat_min=13.0,lon_max=245.0, lat_max=32.0

#--------------------------------
# pseudocolor (pcolor) plot on unstructured grid
#--------------------------------

def ug_pcolor_cell(
        axis = None,
        data = None,
        vertexid = np.nan,
        xvertex = np.nan,
        yvertex = np.nan,
        nedges_cell = np.nan,
        vertices_cell = np.nan,
        **kwargs,
        ):
    """Pseudocolor plot on unstructured grid (cells)

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :data:          (array-like) data to plot
    :vertexid:      (array-like) vertex ID
    :xvertex:       (array-like) x-coordinate of vertices
    :yvertex:       (array-like) y-coordinate of vertices
    :nedges_cell:   (array-like) number of edges on cells
    :vertices_cell: (array-like) vertices on cells
    :**kwargs:      (keyword arguments, optional) passed along to the PatchCollection constructor
    :return:        (matplotlib.collections.PatchCollection)

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # patches
    patches = []
    ncell = nedges_cell.size
    for i in np.arange(ncell):
        vid = vertices_cell[i,:nedges_cell[i]]
        vidx = vid-1
        # TODO: assuming the vertex id is the index + 1 for now
        # which is much faster than the following code using vertexid
        # vidx = np.zeros(vid.size, np.int)
        # for j in np.arange(vid.size):
        #     vidx[j] = np.argwhere(vertexid==vid[j])
        xp = xvertex[vidx]
        yp = yvertex[vidx]
        patches.append(Polygon(list(zip(xp,yp))))
    # plot patch collection
    pc = PatchCollection(patches, **kwargs)
    if data is not None:
        pc.set_array(data)
    fig = axis.add_collection(pc)
    return fig

def ug_pcolor_vertex(
        axis = None,
        data = None,
        cellid = np.nan,
        xcell = np.nan,
        ycell = np.nan,
        cells_vertex = np.nan,
        **kwargs,
        ):
    """Pseudocolor plot on unstructured grid (dual cells centered on vertices)

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :data:          (array-like) data to plot
    :cellid:        (array-like) cell ID
    :xcell:         (array-like) x-coordinate of cells
    :ycell:         (array-like) y-coordinate of cells
    :cells_vertex:  (array-like) cells on vertices
    :**kwargs:      (keyword arguments, optional) passed along to the PatchCollection constructor
    :return:        (matplotlib.collections.PatchCollection)

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # patches
    patches = []
    idx_mask = []
    nvertex = cells_vertex.shape[0]
    for i in np.arange(nvertex):
        cid = cells_vertex[i,:]
        cidx = cid-1
        # TODO: assuming the cell id is the index + 1 for now
        # which is much faster than the following code using cellid
        # cidx = np.zeros(cid.size, np.int)
        # for j in np.arange(cid.size):
        #     cidx[j] = np.argwhere(cellid==cid[j])
        if any(cidx == -1):
            idx_mask.append(i)
            continue
        xp = xcell[cidx]
        yp = ycell[cidx]
        patches.append(Polygon(list(zip(xp,yp))))
    data = np.delete(data, idx_mask)
    # plot patch collection
    pc = PatchCollection(patches, **kwargs)
    if data is not None:
        pc.set_array(data)
    fig = axis.add_collection(pc)
    return fig

def ug_pcolor_edge(
        axis = None,
        data = None,
        edgeid = np.nan,
        xvertex = np.nan,
        yvertex = np.nan,
        vertices_edge = np.nan,
        **kwargs,
        ):
    """Pseudocolor plot on unstructured grid (edges)

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :data:          (array-like) data to plot
    :edgeid:        (array-like) edge ID
    :xvertex:       (array-like) x-coordinate of vertices
    :yvertex:       (array-like) y-coordinate of vertices
    :vertices_edge: (array-like) vertices on edges
    :**kwargs:      (keyword arguments, optional) passed along to the LineCollection constructor
    :return:        (matplotlib.collections.LineCollection)

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # patches
    segments = []
    nedge = vertices_edge.shape[0]
    for i in np.arange(nedge):
        vid = vertices_edge[i,:]
        vidx = vid-1
        # TODO: assuming the edge id is the index + 1 for now
        # which is much faster than the following code using edgeid
        # vidx = np.zeros(vid.size, np.int)
        # for j in np.arange(vid.size):
        #     vidx[j] = np.argwhere(edgeid==vid[j])
        xp = xvertex[vidx]
        yp = yvertex[vidx]
        segments.append(list(zip(xp,yp)))
    segments = np.array(segments)
    # plot line collection
    lc = LineCollection(segments, **kwargs)
    if data is not None:
        lc.set_array(data)
    fig = axis.add_collection(lc)
    return fig

def ug_pcolor_cell_periodic(
        axis = None,
        data = None,
        xperiod = 0,
        yperiod = 0,
        vertexid = np.nan,
        xvertex = np.nan,
        yvertex = np.nan,
        xcell = np.nan,
        ycell = np.nan,
        dv_edge = np.nan,
        nedges_cell = np.nan,
        vertices_cell = np.nan,
        **kwargs,
        ):
    """Pseudocolor plot on unstructured grid (cells, periodic)

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :data:          (array-like) data to plot
    :vertexid:      (array-like) vertex ID
    :xperiod:       (float) period in x-direction
    :yperiod:       (float) period in y-direction
    :xvertex:       (array-like) x-coordinate of vertices
    :yvertex:       (array-like) y-coordinate of vertices
    :xcell:         (array-like) x-coordinate of cells
    :ycell:         (array-like) y-coordinate of cells
    :dv_edge:       (array like) length of edges, distance between vertices on edge
    :nedges_cell:   (array-like) number of edges on cells
    :vertices_cell: (array-like) vertices on cells
    :**kwargs:      (keyword arguments, optional) passed along to the PatchCollection constructor
    :return:        (matplotlib.collections.PatchCollection)

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # maximum edge length
    dv_edge_small = 1.0
    dv_edge_max = dv_edge.max() + dv_edge_small
    # patches
    patches = []
    ncell = nedges_cell.size
    for i in np.arange(ncell):
        vid = vertices_cell[i,:nedges_cell[i]]
        vidx = vid-1
        # TODO: assuming the vertex id is the index + 1 for now
        # which is much faster than the following code using vertexid
        # vidx = np.zeros(vid.size, np.int)
        # for j in np.arange(vid.size):
        #     vidx[j] = np.argwhere(vertexid==vid[j])
        xp = xvertex[vidx]
        yp = yvertex[vidx]
        if any(np.abs(xp[0:-1]-xp[1:]) > dv_edge_max) or \
           any(np.abs(yp[0:-1]-yp[1:]) > dv_edge_max):
            xc = xcell[i]
            yc = ycell[i]
            for j in np.arange(nedges_cell[i]):
                if xp[j] - xc > dv_edge_max:
                    xp[j] -= xperiod
                elif xp[j] - xc < -dv_edge_max:
                    xp[j] += xperiod
                if yp[j] - yc > dv_edge_max:
                    yp[j] -= yperiod
                elif yp[j] - yc < -dv_edge_max:
                    yp[j] += yperiod
        patches.append(Polygon(list(zip(xp,yp))))
    # plot patch collection
    pc = PatchCollection(patches, **kwargs)
    if data is not None:
        pc.set_array(data)
    if len(patches) > 64:
        pc.set_linewidth(0.1)
    fig = axis.add_collection(pc)
    return fig

def ug_pcolor_vertex_periodic(
        axis = None,
        data = None,
        xperiod = 0,
        yperiod = 0,
        cellid = np.nan,
        xvertex = np.nan,
        yvertex = np.nan,
        xcell = np.nan,
        ycell = np.nan,
        dv_edge = np.nan,
        cells_vertex = np.nan,
        **kwargs,
        ):
    """Pseudocolor plot on unstructured grid (dual cells centered on vertices, periodic)

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :data:          (array-like) data to plot
    :cellid:        (array-like) cell ID
    :xperiod:       (float) period in x-direction
    :yperiod:       (float) period in y-direction
    :xvertex:       (array-like) x-coordinate of vertices
    :yvertex:       (array-like) y-coordinate of vertices
    :xcell:         (array-like) x-coordinate of cells
    :ycell:         (array-like) y-coordinate of cells
    :dv_edge:       (array like) length of edges, distance between vertices on edge
    :cells_vertex:  (array-like) cells on vertices
    :**kwargs:      (keyword arguments, optional) passed along to the PatchCollection constructor
    :return:        (matplotlib.collections.PatchCollection)

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # maximum edge length
    dv_edge_small = 1.0
    dv_edge_max = dv_edge.max() + dv_edge_small
    # patches
    patches = []
    idx_mask = []
    nvertex = cells_vertex.shape[0]
    for i in np.arange(nvertex):
        cid = cells_vertex[i,:]
        cidx = cid-1
        # TODO: assuming the cell id is the index + 1 for now
        # which is much faster than the following code using cellid
        # cidx = np.zeros(cid.size, np.int)
        # for j in np.arange(cid.size):
        #     cidx[j] = np.argwhere(cellid==cid[j])
        if any(cidx == -1):
            idx_mask.append(i)
            continue
        xp = xcell[cidx]
        yp = ycell[cidx]
        if any(np.abs(xp[0:-1]-xp[1:]) > dv_edge_max) or \
           any(np.abs(yp[0:-1]-yp[1:]) > dv_edge_max):
            xc = xvertex[i]
            yc = yvertex[i]
            for j in np.arange(3):
                if xp[j] - xc > dv_edge_max:
                    xp[j] = xp[j] - xperiod
                elif xp[j] - xc < -dv_edge_max:
                    xp[j] = xp[j] + xperiod
                if yp[j] - yc > dv_edge_max:
                    yp[j] = yp[j] - yperiod
                elif yp[j] - yc < -dv_edge_max:
                    yp[j] = yp[j] + yperiod
        patches.append(Polygon(list(zip(xp,yp))))
    data = np.delete(data, idx_mask)
    # plot patch collection
    pc = PatchCollection(patches, **kwargs)
    if data is not None:
        pc.set_array(data)
    if len(patches) > 64:
        pc.set_linewidth(0.1)
    fig = axis.add_collection(pc)
    return fig

def ug_pcolor_edge_periodic(
        axis = None,
        data = None,
        xperiod = 0,
        yperiod = 0,
        edgeid = np.nan,
        xvertex = np.nan,
        yvertex = np.nan,
        xedge = np.nan,
        yedge = np.nan,
        dv_edge = np.nan,
        vertices_edge = np.nan,
        **kwargs,
        ):
    """Pseudocolor plot on unstructured grid (edges, periodic)

    :axis:          (matplotlib.axes, optional) axis to plot the figure on
    :data:          (array-like) data to plot
    :xperiod:       (float) period in x-direction
    :yperiod:       (float) period in y-direction
    :edgeid:        (array-like) edge ID
    :xvertex:       (array-like) x-coordinate of vertices
    :yvertex:       (array-like) y-coordinate of vertices
    :xedge:         (array-like) x-coordinate of edges
    :yedge:         (array-like) y-coordinate of edges
    :dv_edge:       (array like) length of edges, distance between vertices on edge
    :vertices_edge: (array-like) vertices on edges
    :**kwargs:      (keyword arguments, optional) passed along to the LineCollection constructor
    :return:        (matplotlib.collections.LineCollection)

    """
    # use curret axis if not specified
    if axis is None:
        axis = plt.gca()
    # maximum edge length
    dv_edge_small = 1.0
    dv_edge_max = dv_edge.max() + dv_edge_small
    # patches
    segments = []
    nedge = vertices_edge.shape[0]
    for i in np.arange(nedge):
        vid = vertices_edge[i,:]
        vidx = vid-1
        # TODO: assuming the edge id is the index + 1 for now
        # which is much faster than the following code using edgeid
        # vidx = np.zeros(vid.size, np.int)
        # for j in np.arange(vid.size):
        #     vidx[j] = np.argwhere(edgeid==vid[j])
        xp = xvertex[vidx]
        yp = yvertex[vidx]
        if any(np.abs(xp[0:-1]-xp[1:]) > dv_edge_max) or \
           any(np.abs(yp[0:-1]-yp[1:]) > dv_edge_max):
            xc = xedge[i]
            yc = yedge[i]
            for j in np.arange(2):
                if xp[j] - xc > dv_edge_max:
                    xp[j] -= xperiod
                elif xp[j] - xc < -dv_edge_max:
                    xp[j] += xperiod
                if yp[j] - yc > dv_edge_max:
                    yp[j] -= yperiod
                elif yp[j] - yc < -dv_edge_max:
                    yp[j] += yperiod
        segments.append(list(zip(xp,yp)))
    segments = np.array(segments)
    # plot line collection
    lc = LineCollection(segments, **kwargs)
    if data is not None:
        lc.set_array(data)
    fig = axis.add_collection(lc)
    return fig

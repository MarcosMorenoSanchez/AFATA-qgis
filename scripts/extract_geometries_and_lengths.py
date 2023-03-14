def extract_geometries_and_lengths(layer, sort_field):
    '''
    Extracts Linestring or Multilinestring geometries and their lengths from a vector layer.

    Parameters:
    ----------
    layer: QgsVectorLayer
        The vector layer to extract the information from.
    sort_field: str
        The field to order the features by.

    Returns:
    ----------
    A tuple of two lists:
        - A list of geometries, where each geometry is a list of x and y coordinates.
        - A list of lengths, where each length corresponds to the geometry at the same index in the geometries list.

    Raises:
    ----------
    ValueError:
        If the layer is not a line vector layer.

    '''

    # Check if the layer is a line vector layer
    if layer.geometryType() != QgsWkbTypes.LineGeometry:
        print('Warning: the layer is not a line vector layer.')
    # create a QgsFeatureRequest object and set the sort order
    request = QgsFeatureRequest().addOrderBy(sort_field, ascending=True)

    # get all features sorted by the specified field
    features = layer.getFeatures(request)

    geometries = []
    lengths =[]
    # loop through the features
    for feature in features:
        attrs = feature.attributes()
        id = feature.id()
        geom = feature.geometry()
        geomSingleType = QgsWkbTypes.isSingleType(geom.wkbType())

        if geom.type() == QgsWkbTypes.LineGeometry:
            if geomSingleType:
                x = geom.asPolyline()
                geometries.append(x)
                lengths.append(geom.length())
            else:
                x = geom.asMultiPolyline()
                for line in x:
                    geometries += [line for line in x]
                    lengths += [geom.length() for line in x]
    return geometries, lengths

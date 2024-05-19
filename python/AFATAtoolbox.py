# -*- coding: utf-8 -*-

"""
***************************************************************************
*                                                                         *
*   AFATAtool.py                                                          *
*                                                                         *
*   Copyright (C) 2023 Marcos Moreno Sánchez                              *
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
*   This program is distributed in the hope that it will be useful,       *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
*   GNU General Public License for more details.                          *
*                                                                         *
*   You should have received a copy of the GNU General Public License     *
*   along with this program; if not, write to the Free Software           *
*   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,            *
*   MA 02110-1301, USA.                                                   *
*                                                                         *
*   For additional information, contact to:                               *
*   Marcos Moreno Sánchez                                                 *
*   marcosmrnsnchz@gmail.com                                              *
*                                                                         *
*   Davide Torre                                                          *
*   Department of Earth Sciences, Sapienza University of Rome             *
*   00145 Roma, Italy                                                     *
*   davide.torre@uniroma1.it                                              *
*                                                                         *
*   Version: 1.0                                                          *
*   May 3, 2023                                                           *
*                                                                         *
*   Last modified May 19, 2024                                            *
*                                                                         *
***************************************************************************
"""

from qgis.PyQt.QtCore import (QCoreApplication,
                              QVariant)

from qgis.core import (QgsProcessing,
                       QgsProcessingException,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterDistance,
                       QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination,
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterVectorDestination,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterField,
                       QgsProcessingParameterFeatureSink,
                       QgsFeatureSink,
                       QgsFeature,
                       QgsVectorLayer,
                       QgsField,
                       QgsFields,
                       QgsWkbTypes,
                       QgsFeatureRequest,
                       QgsGeometry,
                       QgsPoint,
                       QgsRectangle,
                       QgsPointXY,
                       QgsEditError,
                       QgsProject)

from qgis.analysis import QgsZonalStatistics

import numpy as np


class AFATAtool(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT_VECTOR_DOWN = 'INPUT_VECTOR_DOWN'
    INPUT_VECTOR_UP = 'INPUT_VECTOR_UP'
    INPUT_SORT_FIELD_VECTOR = 'INPUT_SORT_FIELD_VECTOR'
    INPUT_RASTER_DEM = 'INPUT_RASTER_DEM'
    INPUT_RASTER_ASPECT = 'INPUT_RASTER_ASPECT'
    INPUT_RASTER_SLOPE = 'INPUT_RASTER_SLOPE'
    INPUT_PARAMETER_SPACE = 'INPUT_PARAMETER_SPACE'
    INPUT_PARAMETER_TRANSECT_SPACE = 'INPUT_PARAMETER_TRANSECT_SPACE'

    INPUT_CSV_PITCH = 'INPUT_CSV_PITCH'
    INPUT_FIELD_NAME_CSV = 'INPUT_FIELD_NAME_CSV'

    OUTPUT_VECTOR_TRANSECT = 'OUTPUT_VECTOR_TRANSECT'
    #OUTPUT_CSV_VECTOR_TRANSECT

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return AFATAtool()

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm.
        """
        return 'afatatoolbox'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('AFATA toolbox')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr('AFATA Toolbox')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to.
        """
        return 'afatagroup'

    def shortHelpString(self):
        """
        Returns a localised short helper string for the algorithm.
        """
        return self.tr(
            "Evaluate the three components of rupture surface (horizontal, "
            "vertical and along plane offsets) of a normal fault, by regularly "
            "scanning with an high number of measurements. "
            "\n-----------------------------------------------------------------"
            "\n See for more details: "
            "\n Torre, D., Moreno-Sánchez, M., Bello, S. and Menichetti, M., 2024."
            "\n AFATA (Active FAult Tectonic Analysis): a semi-automatic tool "
            "for QGIS for estimating fault offsets on superficial ruptures."
            # "\n Comput. Geosci. doi:xx.xxxx/x.xxxxx.xxxx.xx.xxx"
        )

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR_DOWN,
                self.tr('Fault bottom edge'),
                types=[QgsProcessing.TypeVectorLine],
                defaultValue=None
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_VECTOR_UP,
                self.tr('Fault top edge'),
                types=[QgsProcessing.TypeVectorLine],
                defaultValue=None
            )
        )

        self.addParameter(
            QgsProcessingParameterField(
                self.INPUT_SORT_FIELD_VECTOR,
                self.tr('Matching segment field'),
                None,
                self.INPUT_VECTOR_DOWN,
                QgsProcessingParameterField.Any
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_CSV_PITCH,
                self.tr('Pitch dataset'),
                types=[QgsProcessing.TypeFile],
                defaultValue=None,
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.INPUT_FIELD_NAME_CSV,
                self.tr('Pitch field'),
                None,
                self.INPUT_CSV_PITCH,
                QgsProcessingParameterField.Any,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterDistance(
                self.INPUT_PARAMETER_SPACE,
                self.tr('Fault segmentation step'),
                defaultValue=0.5,
                # Make distance units match the INPUT layer units:
                parentParameterName=self.INPUT_VECTOR_DOWN
            )
        )
        self.addParameter(
            QgsProcessingParameterDistance(
                self.INPUT_PARAMETER_TRANSECT_SPACE,
                self.tr('Join segment spacing'),
                defaultValue=0.1,
                # Make distance units match the INPUT layer units:
                parentParameterName=self.INPUT_VECTOR_DOWN
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER_DEM,
                self.tr('Digital Elevation Model [DEM]'),
                defaultValue=None
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER_ASPECT,
                self.tr('Aspect raster'),
                defaultValue=None
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_RASTER_SLOPE,
                self.tr('Slope raster'),
                defaultValue=None
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_VECTOR_TRANSECT,
                self.tr('Output AFATA transect'),
                type=QgsProcessing.TypeVectorLine
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        def GenerateInterpolatedPoints(lineGeometry, lengthLine, distanceInterpolation):
            """
            Generate interpolated points with a specified distance through line geometry.

            Parameters:
            ----------
            lineGeometry : QgsGeometry
                The line geometry to extract the information from.

            lengthLine : float
                The total length of the line.

            distanceInterpolation : float
                The distance at which to subdivide the line into points.

            Returns:
            ----------
            list of QgsGeometry
                A list of point geometries interpolated or nearest to the specified distances along the lineGeometry.
            """

            # Calculate the distances at which each point will be located
            distancesPoints = [distanceInterpolation *(1 + n) for n in range(int(lengthLine / distanceInterpolation))]
            # For each point in numberPoints, we get its coordinate on the lineGeometry, following its distance
            positionPoints = [lineGeometry.interpolate(distance) for distance in distancesPoints]
    
            # Returns a list with QgsPoints of the line
            return positionPoints

        def FindNearestPointsOnLine(lineGeometry, pointsList):
            """
            Find the nearest point on a line geometry for each point in a list.

            Parameters:
            ----------
            
            lineGeometry: QgsGeometry
                The line geometry to find nearest points on.
                
            pointsList: list of QgsPoint
                A list of points to find the nearest points for, on the line.

            Returns:
            ----------
            list of QgsGeometry
                A list of geometries representing the nearest points on the line.
            """

            positionPoints = [lineGeometry.nearestPoint(point) for point in pointsList]

            return positionPoints
        
        def CreatePolylinesFromPointPairs (pointListA, pointListB):
            """
            Creation of polylines from two lists of points.

            Parameters:
            ----------
            pointListA: list of QgsGeometry
                A list of point geometries from line A.

            pointListB: list of QgsGeometry
                A list of points geometries from line B.

            Returns:
            ----------
            list of QgsGeometry
                A list of geometries representing polylines that merge each corresponding point from list A to B.
            """

            polylines = [QgsGeometry.fromPolylineXY([a.asPoint(), b.asPoint()]) for a, b in zip(pointListA, pointListB)]

            return polylines

        def CalculateZonalStatistics(buffers, rasterLayer, nameColumn):
            """
            Calculate mean zonal statistics for a list of buffer geometries over a given raster layer.

            Parameters:
            ----------
            buffers: list of QgsGeometry
                List of buffer geometries.
            rasterLayer: QgsRasterLayer
                The raster layer from which to calculate zonal statistics.
            nameColumn: string
                A string to name a column where data will be stored in the temporary layer.

            Returns:
            ----------
            list
                A list of mean values from the zonal statistics calculation.
            """
            # Obtain the CRS from the rasterLayer
            crs = rasterLayer.crs().authid()

            # Create a temporary polygon layer for the buffers with the same CRS as the rasterLayer
            tempLayer = QgsVectorLayer(f'Polygon?crs={crs}', 'polygon_temp', 'memory')
            prov = tempLayer.dataProvider()

            # Add a new feature for each buffer and assign the geometry
            #prov.addFeatures([QgsFeature(geometry=bufferItem) for bufferItem in buffers])
            #tempLayer.updateExtents() 
            # Update the layer's extents after adding new features 
            # to ensure the layer's spatial properties accurately reflect its contents.
            # This is crucial for correct visualization and spatial analysis within QGIS, 
            # as it recalculates the bounding box to include all current features.
            features = []
            for bufferItem in buffers:
                feature = QgsFeature()
                feature.setGeometry(bufferItem)  # Asignar la geometría usando setGeometry
                features.append(feature)

            prov.addFeatures(features)
            tempLayer.updateExtents()
            # Calculate the zonal statistics
            zone_stat = QgsZonalStatistics(tempLayer, rasterLayer, nameColumn, 1, QgsZonalStatistics.Mean)
            zone_stat.calculateStatistics(None)

            # Retrieve the zonal statistics results
            rasterAverage = []
            for feature in tempLayer.getFeatures():
                attrs = feature.attributes()
                rasterAverage.append(attrs[-1])  # Assuming the mean value is the last attribute

            return rasterAverage

        def GetIntersectionPointOfLines(x1, y1, x2, y2, x3, y3, x4, y4):
            """ :: Credits for this algorithm to PaulBrocks1988
            Intersection point of two line segments in 2 dimensions

            params:
            ----------
            x1, y1, x2, y2 -> coordinates of line a, p1 ->(x1, y1), p2 ->(x2, y2),

            x3, y3, x4, y4 -> coordinates of line b, p3 ->(x3, y3), p4 ->(x4, y4)

            Return:
            ----------
            list
                A list contains x and y coordinates of the intersection point,
                but return an empty list if no intersection point.

            """
            # None of lines' length could be 0.
            if ((x1 == x2 and y1 == y2) or (x3 == x4 and y3 == y4)):
                return []

            # The denominators for the equations for ua and ub are the same.
            den = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

            # Lines are parallel when denominator equals to 0,
            # No intersection point
            if den == 0:
                return []

            # Avoid the divide overflow
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (den + 1e-16)
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / (den + 1e-16)

            # Return a list with the x and y coordinates of the intersection
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)

            return QgsPoint(x, y)

        def ExtractLinestringsAndLengths(vectorLayer, sortField):
            ''' 
            Extracts Linestring or Multilinestring geometries and their lengths from a vector layer.

            Parameters:
            ----------
            vectorLayer: QgsVectorLayer
                The vector layer to extract the information from.
            sortField: string
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

            # Create a QgsFeatureRequest object and set the sort order
            request = QgsFeatureRequest().addOrderBy(sortField, ascending=True)

            # Get all features sorted by the specified field
            features = vectorLayer.getFeatures(request)

            lineGeometries = []
            lineLengths = []
            # Loop through the features
            for feature in features:
                attrs = feature.attributes()
                id = feature.id()
                geom = feature.geometry()
                lineGeometries.append(geom)
                lineLengths.append(geom.length())
            return lineGeometries, lineLengths

        def GetIntersectingEdgePoints(pointsList, rasterAverageList, lineGeometry2Intersect, countLine):
            """ 
            Get edge points of a line segment that intersect with each given point.

            Parameters:
            -----------
            pointsList: list of QgsPoint
                The list of points to check for intersections with the line segment.

            rasterAverageList: list of float
                The list of aspect mean values for each point.

            lineGeometry2Intersect: QgsGeometry
                The QgsGeometry object representing the line segment.

            countLine: int
                The index of the line segment.

            Returns:
            --------
            list of QgsPoint
                A list of intersection points.
            """

            # Get total of vertex at line to intersect
            vertexs = lineGeometry2Intersect[countLine].asMultiPolyline()
            for v in vertexs:
                lineVertex = len(v)

            edgePoints = []

            for j, point in enumerate(pointsList):

                # Segment az: use rasterAverageList of centroids to calculate a line the is create by fbePoints and an azimuth
                segment_az = [point.asPoint().x(), point.asPoint().y(),
                              point.asPoint().x() + np.sin(np.radians(rasterAverageList[j])),
                              point.asPoint().y() + np.cos(np.radians(rasterAverageList[j]))]

                # Segment a: is the line that join the (i)-vertex with the next (i+1)-vertex in up layer
                for i in range(lineVertex - 1):
                    segment_a = [lineGeometry2Intersect[countLine].vertexAt(i).x(), lineGeometry2Intersect[countLine].vertexAt(i).y(),
                                 lineGeometry2Intersect[countLine].vertexAt(i + 1).x(), lineGeometry2Intersect[countLine].vertexAt(i + 1).y()]

                    # We check if segment_az and segment_a have a common point.
                    # If this is not the case, the loop continues with a new segment_a.
                    # On the other hand, if there is an intersection, it is checked whether
                    # it is contained within the rectangle defined by the upper line vertices.
                    try:
                        ei = GetIntersectionPointOfLines(*segment_a, *segment_az)

                        if ei.boundingBoxIntersects(
                                QgsRectangle(lineGeometry2Intersect[countLine].vertexAt(i).x(), lineGeometry2Intersect[countLine].vertexAt(i).y(),
                                             lineGeometry2Intersect[countLine].vertexAt(i + 1).x(), lineGeometry2Intersect[countLine].vertexAt(i + 1).y())):
                            edgePoints.append(ei)
                            break

                    # If an error occurred is print to user.
                    except Exception as e:
                        print("An error occurred: ", e)
                else:
                    j += 1

            return edgePoints
        
        def CalculateElevationDifferences(pointListA, pointListB, rasterLayer):
            """
            Calculate the elevation differences between pairs of start and end points using a DEM raster layer.
        
            Parameters:
            ----------
            pointListA: list of QgsPointXY
                List of starting points.
            pointListB: list of QgsPointXY
                List of ending points.
            rasterLayer: QgsRasterLayer
                The DEM raster layer to extract elevation data from.
        
            Returns:
            ----------
            list
                A list of elevation differences (absolute values) between each pair of start and end points.
            """
            rv = []  # List to store the elevation differences
        
            # Ensure that the lists of points have the same length
            if len(pointListA) != len(pointListB):
                raise ValueError("The lists of start and end points must have the same length.")
        
            for d_pto, u_pto in zip(pointListA, pointListB):
                # Extract elevation for the start and end point
                z0_result = rasterLayer.dataProvider().sample(d_pto.asPoint(), 1)
                z1_result = rasterLayer.dataProvider().sample(u_pto.asPoint(), 1)
        
                # Check if the sampling was successful
                if z0_result[1] and z1_result[1]:
                    z0 = z0_result[0]
                    z1 = z1_result[0]
                    # Calculate the absolute elevation difference
                    dh = abs(z1 - z0)
                    rv.append(dh)
                else:
                    # Handle cases where elevation data could not be extracted
                    rv.append(None)  # Or consider other ways to handle these cases
        
            return rv
        
        def CalculateAzimuth(point1, point2):
            """
            Calculate the azimuth direction between two points given.
            """
            dx = point2.x() - point1.x()
            dy = point2.y() - point1.y()
            azimuth = np.arctan2(dx, dy) # North-Clockwise Convention
            azimuth_degrees = np.degrees(azimuth)
            if azimuth_degrees < 0:
                azimuth_degrees += 360
            return azimuth_degrees

        #############################################################################################
        #############################################################################################
        ####                                                                                    #####
        ####                            Main Script of AFATA tool                               #####
        ####                                                                                    #####
        #############################################################################################
        #############################################################################################
        
        ####                                                                                    #####
        ####                            1) Get all input layers from user                       #####
        ####                                                                                    #####

        # Vector layers and name of the field that match fault lines vector's and pitch field
        # Fault bottom edge == fbe
        # Fault top edge == fte
        # Matching segment field == sortField 
        # Pitch dataset == csvLayer
        # Pitch field == pitchField
        # Fault segmentation step == fssDistance
        # Join segment spacing == jssTransectDistance
        

        fbeLayer = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR_DOWN, context)
        fteLayer = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR_UP, context)
        sortField = self.parameterAsString(parameters, self.INPUT_SORT_FIELD_VECTOR, context)

        # Upload file csv with pitch angle
        csvLayer = self.parameterAsVectorLayer(parameters, self.INPUT_CSV_PITCH, context)
        pitchField = self.parameterAsString(parameters, self.INPUT_FIELD_NAME_CSV, context)

        # Raster Layer
        demLayer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER_DEM, context)
        aspectLayer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER_ASPECT, context)
        slopeLayer = self.parameterAsRasterLayer(parameters, self.INPUT_RASTER_SLOPE, context)

        # User's value for interpolation
        fssDistance = self.parameterAsDouble(parameters, self.INPUT_PARAMETER_SPACE, context)
        jssTransectDistance = self.parameterAsDouble(parameters, self.INPUT_PARAMETER_TRANSECT_SPACE, context)

        ####                                                                                    #####
        ####                            2) Abort execution when layer upload fails              #####
        ####                                                                                    #####

        # Report if any parameter is not valid
        if not fbeLayer.isValid():
            feedback.reportError('\nERROR: Fault bottom edge is not valid.\n')
            return {}
        if not fteLayer.isValid():
            feedback.reportError('\nERROR: Fault top edge is not valid.\n')
            return {}
        if not demLayer.isValid():
            feedback.reportError('\nERROR: DEM raster layer is not valid.\n')
            return {}
        if not aspectLayer.isValid():
            feedback.reportError('\nERROR: Aspect raster layer is not valid.\n')
            return {}
        if not slopeLayer.isValid():
            feedback.reportError('\nERROR: Slope raster layer is not valid.\n')
            return {}

        # Faults shapefiles must have same quantity of features
        if fbeLayer.featureCount() != fteLayer.featureCount():
            feedback.reportError('\nERROR: Vector layers -Fault bottom edge- and -Fault top edge- must have the same number of features.\n')
            return {}

        ####                                                                                    #####
        ####                            3) Preparation of output vector file                    #####
        ####                                                                                    #####

        # Creation of Output vector layer (Transect's layer)
        outFields = QgsFields()
        outFields.append(QgsField("LineID", QVariant.Int))
        outFields.append(QgsField("FID", QVariant.Int))
        outFields.append(QgsField("SLP_MAX", QVariant.Double))
        outFields.append(QgsField("SLP", QVariant.Double))
        outFields.append(QgsField("SLP_MIN", QVariant.Double))
        outFields.append(QgsField("RV", QVariant.Double))
        outFields.append(QgsField("RH", QVariant.Double))
        outFields.append(QgsField("RP_MAX", QVariant.Double))
        outFields.append(QgsField("RP", QVariant.Double))
        outFields.append(QgsField("RP_MIN", QVariant.Double))

        # In case that user have csvLayer for pitch layer, an extra field will create on the output layer
        if csvLayer == None or csvLayer == "":
            #If csvLayer is not given, then is not necesary to create another field.
            pass
        else:
            # Create a QgsFeatureRequest object and set the sort order
            request = QgsFeatureRequest().addOrderBy(sortField, ascending=True)

            # Get all features sorted by the specified field
            features = csvLayer.getFeatures(request)

            pitchValues = [float(feature[pitchField]) for feature in features]
            # Buffer value to get strike-direction along Fault top edge
            fteBufferRadius = 0.1 
            outFields.append(QgsField("PITCH", QVariant.String))
            pass
        
        #Here we will create a destination path ('sink') to save later all our result. There is definition of 
        #output fields, type of geometry that will be use and the CRS that will use the layer.
        (sink, destID) = self.parameterAsSink(parameters, self.OUTPUT_VECTOR_TRANSECT, context,
                                               outFields, QgsWkbTypes.LineString, fbeLayer.sourceCrs())

        ####                                                                                    #####
        ####                            4) Set up elements for AFATA workflow                   #####
        ####                                                                                    #####

        # Obtention of geometries of vector layers and order by field defined by user, both layer must have the same field name

        fteGeometry, fteLength = ExtractLinestringsAndLengths(fteLayer, sortField)
        fbeGeometry, fbeLength = ExtractLinestringsAndLengths(fbeLayer, sortField)

        # AFATA transcet will be calculate for each line segment of the total lines given by fteLength
        for line in range(len(fteLength)):
            
            ####                                                                                    #####
            ####                            5) Retrieve all parameters to calculate transects       #####
            ####                                                                                    #####

            fbePoints = GenerateInterpolatedPoints(fbeGeometry[line], fbeLength[line], fssDistance)
            ftePoints = FindNearestPointsOnLine(fteGeometry[line], fbePoints)
            pointPairsPolylines = CreatePolylinesFromPointPairs(fbePoints, ftePoints)
            pointPairsPolylinesLengths = [line.length() for line in pointPairsPolylines]
            centroids = [line.centroid() for line in pointPairsPolylines]
            buffers = [centroid.buffer(distance=length / 2, segments=32) for centroid, length in zip(centroids, pointPairsPolylinesLengths)]            
            aspectMeanValues = CalculateZonalStatistics(buffers, aspectLayer, nameColumn='Aspect-')
            intersectingEdgePoints = GetIntersectingEdgePoints(fbePoints,aspectMeanValues,fteGeometry,line)

            ####                                                                                    #####
            ####                            6.1) Calculations with Pitch Angle                      #####
            ####                                                                                    #####
            # Next step is related to pitch calculations, if user attach a csvPitch layer.
            if not csvLayer or pitchValues[line] == 0.0:
                # If not csvLayer has been given by user or pitchValue is 0, will continue.
                pass
            else:
                # Creation of empty list for storage values
                lineStrikeList, azimuthPitchList, azimuthList, euclideanDistanceList= [],[],[],[]

                # Get intersection line in fteGeometry with intersectionEdgePoints 
                for point in intersectingEdgePoints:
                    # First, get x and y form each intersectingEdgePoints and transform to geometry of PointXY
                    gPoint = QgsGeometry.fromPointXY(QgsPointXY(point.x(),point.y()))
                    # Second, creation of buffer around the gPoint
                    bff = gPoint.buffer(fteBufferRadius, 32)
                    # Third, get all vertex from fteGeometry that intersect with gPoint buffer
                    lineStrike = fteGeometry[line].intersection(bff)
                    lineStrikeList.append(lineStrike)

                # From previous step we will get calculation of azimuths for each vertex for each lineStrike
                for i in range(len(lineStrikeList)):
                    
                    # Open the i element of lineStrike form lineStrikeList as a polyline, which allow us to extract vertex points information
                    lineStrikePoints = lineStrikeList[i].asPolyline()
                    
                    # Calculation distance and azimuth in the polyline for each vertex
                    for i in range(len(lineStrikePoints) - 1):
                        # Get vertex i and the consecutive vertex i+1
                        startPoint = lineStrikePoints[i]
                        endPoint = lineStrikePoints[i+1]
                        
                        # Get Eucledian distance between vertexs and storage in a list
                        dx = startPoint.x() - endPoint.x()
                        dy = startPoint.y() - endPoint.y()
                        euclideanDistance = np.sqrt(dx**2 + dy**2)
                        euclideanDistanceList.append(euclideanDistance)
                        
                        # Get the azimuth value between vertexs in clockwise direction from north and storage in alist
                        azimuthValue = CalculateAzimuth(startPoint,endPoint)
                        azimuthList.append(azimuthValue)
                    
                    # Set total distance of segment
                    totalDistance = sum(euclideanDistanceList)
                    weightDistanceList = [euclideanElement/totalDistance for euclideanElement in euclideanDistanceList]
                    # Calculation of mean azitmuh value using a weight by distance. Distance value has influence in azimuth value.
                    azimuthMeanValue = sum(azimuthElement * weightDistance for azimuthElement, weightDistance in zip(azimuthList, weightDistanceList))
                    # Add pitch angle to azimuthMeanValue in clockwise. With modulus (360) we get azimuth range between [0,360)
                    azimuthPitch = (azimuthMeanValue + pitchValues[line]) % 360
                    azimuthPitchList.append(azimuthPitch)

                # The points outside range are removed
                if len(fbePoints) > len(intersectingEdgePoints):
                    for i in range(len(fbePoints) - len(intersectingEdgePoints)):
                        del fbePoints[-1]
                        del ftePoints[-1]

                # Finally, we recalculate intersectionEdgePoints with the new azimuthPitchList
                intersectingEdgePoints = GetIntersectingEdgePoints(fbePoints,azimuthPitchList,fteGeometry,line)
                pass
            
            ####                                                                                    #####
            ####                            6.2) Next 6.1 OR Calculations without Pitch             #####
            ####                                                                                    #####
            
            # The points outside range are removed
            if len(fbePoints) > len(intersectingEdgePoints):
                for i in range(len(fbePoints) - len(intersectingEdgePoints)):
                    del fbePoints[-1]
                    del ftePoints[-1]

            polylineTransect, slopePointList, transectIds = [], [], []

            # Creation of polyline transect with fbePoints and intersectingEdgePoints
            for i in range(len(fbePoints)):
                #IntersectingEdgePoints are convert to geometry QgsPointXY, and then it is create QgsPolylineXY
                polylineGeometry = QgsGeometry.fromPolylineXY([fbePoints[i].asPoint(), QgsPointXY(intersectingEdgePoints[i])])
                polylineTransect.append(polylineGeometry)

                # Points will be create along each transect with a step distance jssTransectDistance
                currentDistance = 0.0
                while currentDistance < polylineGeometry.length():
                    slopePointElement = polylineTransect[i].interpolate(currentDistance)
                    slopePointList.append(slopePointElement)
                    currentDistance += jssTransectDistance
                    # Each point has a new variable which is set to identificate the point and the transect where its belong
                    transectIds.append(i + 1)

            # All points extract the slope value for slopeLayer
            slopeValuesList = [slopeLayer.dataProvider().sample(point.asPoint(), 1)[0] for point in slopePointList]
            
            # The parameter RV is the calculation of elevations diferrences between point on fbe and fte
            rv = CalculateElevationDifferences(pointListA=fbePoints, pointListB=ftePoints, rasterLayer=demLayer)

            maxs, means, mins = [], [], []

            # For each transect, we will calculate average value of slope using only the points retrieve before.
            for i in range(transectIds[0], max(transectIds) + 1):
                
                auxArray = [slopeValueElement for j, slopeValueElement in enumerate(slopeValuesList) if transectIds[j] == i]

                # auxArray is a list, which has to be converted to an array to perform calculation better
                auxArray = np.array(auxArray)
                avrg = np.mean(auxArray)
                
                # If transect has less 1 points the calculation for maxs or mins average is just the average values, and advert will be appered in screen to user
                if len(auxArray) == 1:
                    means += [avrg]
                    maxs += [avrg] 
                    mins += [avrg]
                    feedback.pushInfo(f"CAUTION!!!\nTransect {i} from line {line+1}: just one point to calculate RP in this segment.\nSuggestion to decrease step distance in transect parameter.\n")
                    pass
                
                # If transect has more than 1 point, then is calculate average, and the average of values that are above (maxs) and below (mins)
                else:
                    means += [avrg]
                    maxs += [np.mean(auxArray[np.where(auxArray > avrg)])]
                    mins += [np.mean(auxArray[np.where(auxArray < avrg)])]
                    pass

            # Then RH is calculate with tangent of average value for simplification
            try:
                rh = [rv[i] / np.tan(np.deg2rad(means[i])) for i in range(len(rv))]
            except IndexError:
                feedback.pushInfo(f"Something unexpected happen")
                raise

            rp, rpMaxs, rpMins = [],[], []

            # Then RP values (maxs, mean, mins) are calculate just from the differents means of slope values
            for i in range(0, max(transectIds)):
                rp += [rv[i] / np.sin(np.deg2rad(means[i]))]
                rpMaxs += [rv[i] / np.sin(np.deg2rad(maxs[i]))]
                rpMins += [rv[i] / np.sin(np.deg2rad(mins[i]))]

            # Finally, all the information is storaged in output file
            for i in range(0, max(transectIds)):
                # Definition of field values
                FID = line + 1
                LINE_ID = i + 1
                SLP_MAX = round(float(maxs[i]), 4)
                SLP = round(float(means[i]), 4)
                SLP_MIN = round(float(mins[i]), 4)
                RV = round(rv[i], 4)
                RH = round(float(rh[i]), 4)
                RP_MAX = round(float(rpMaxs[i]), 4)
                RP = round(float(rp[i]), 4)
                RP_MIN = round(float(rpMins[i]), 4)
                PITCH = None

                # In case that user input a csvLayer, PITCH field will be storage
                if not csvLayer or pitchValues[line] in ["", None, 0.0]:
                    PITCH = "No pitch angle" if not csvLayer else "No pitch field"
                else:
                    PITCH = "Yes pitch angle of " + str(pitchValues[line]) + " degrees"

                # Get the output columns for the line to the outputfile
                outFeat = QgsFeature(outFields)
                geometry = polylineTransect[i]
                outFeat.setGeometry(geometry)

                # Setting the attributes to the output fields
                outFeat.setAttributes([
                    FID,
                    LINE_ID,
                    SLP_MAX,
                    SLP,
                    SLP_MIN,
                    RV,
                    RH,
                    RP_MAX,
                    RP,
                    RP_MIN,
                    PITCH
                ])

                # Final step, add the feature to the output file
                sink.addFeature(outFeat, QgsFeatureSink.FastInsert)

            feedback.pushInfo(f"\nLine {line + 1} of {len(fteLength)}.")
            feedback.pushInfo(f"\n {int(line * 100 / len(fteLength))} % \n")

        results = {self.OUTPUT_VECTOR_TRANSECT: destID}
        return results

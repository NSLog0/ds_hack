import os

from flask import Flask, jsonify
# from flask_cors import CORS, cross_origin
from flask import request
from flask_restplus import Resource, Api, fields,reqparse
from flask_cors import CORS, cross_origin
import hashlib
import json

from .models.ai import Predictor as mypd


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.SWAGGER_UI_DOC_EXPANSION = 'list'
    api = Api(app, version='1.0', title='Sample API',
              description='A sample API')
    cors = CORS(app, resources={r"/*": {"origins": "*"}})
    app.config.from_mapping(
        SECRET_KEY='dev'
    )

    api_version = '/api/v1'

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    app.config['CORS_HEADERS'] = 'application/json'

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    parser = reqparse.RequestParser()
    parser.add_argument('TESTID',        help='ID of RFT tests', location='form')
    parser.add_argument('WellID',        help='ID of Wells', location='form')
    parser.add_argument('DepthMD',       type=float,  help='RFT testing depth (along hole)', location='form')
    parser.add_argument('DepthTVDSS',    type=float,  help='RFT testing depth (true vertical depth subsea)', location='form')
    parser.add_argument('Temp',          type=float,  help='Reservoir Temperature', location='form')
    parser.add_argument('GR',            type=float,  help='Gamma Ray sensor reading', location='form')
    parser.add_argument('Resist_deep',   type=float,  help='Resistivity sensor reading at deep radius of investigation', location='form')
    parser.add_argument('Resist_medium', type=float,  help='Resistivity sensor reading at medium radius of investigation', location='form')
    parser.add_argument('Resist_short',  type=float,  help='Resistivity sensor reading at shallow radius of investigation', location='form')
    parser.add_argument('Density',      type=float,  help='Density sensor reading', location='form')
    parser.add_argument('Neutron',       type=float,  help='Neutron sensor reading', location='form')
    # parser.add_argument('FluidType',     help='Fluid containing in reservoir at specific depth from interpretation of 1 st well log', location='form')
    # parser.add_argument('Subblock',      help='Subsurface block categorized by geological structure', location='form')
    parser.add_argument('Thickness',     type=float,    help='Reservoir Thickness', location='form')
    parser.add_argument('Reservior',     help='Reservoir Name', location='form')

    @cross_origin(origin='*')
    @api.route('{}/predictions'.format(api_version), methods=["post"])
    class Predictions(Resource):
        @api.expect(parser)
        def post(self):
            args = parser.parse_args()
            t = json.loads([ x for x in request.values.items()][0][0])

            acc = mypd.predict(json.dumps(t))
            is_normal = 'OTHER'
            fluid_type = 'Not oil'

            if acc[1] == 1:
                is_normal = 'NORMAL'
            if acc[0] ==  1:
                fluid_type = 'Oil'

            return { "meta": { "code": 200, "message": "success" }, "data": {
                'RFT': is_normal, 'fluid_type': fluid_type,'mobility_score':
                acc[2] } }


    return app

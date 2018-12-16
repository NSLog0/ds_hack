import os

from flask import Flask, jsonify
# from flask_cors import CORS, cross_origin
from flask import request
from flask_restplus import Resource, Api, fields
import hashlib
import json

from .models.ai import Predictor as mypd


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.SWAGGER_UI_DOC_EXPANSION = 'list'
    api = Api(app, version='1.0', title='Sample API',
              description='A sample API')
    # cors = CORS(app, resources={r"/*": {"origins": "*"}})
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

    # app.config['CORS_HEADERS'] = 'application/json'

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    parser = api.parser()
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

    @api.route('{}/predictions'.format(api_version))
    class Predictions(Resource):
        @api.expect(parser)
        def post(self):
            args = parser.parse_args()
            # import ipdb; ipdb.set_trace()
            acc = mypd.predict(json.dumps(args))
            is_nomal = 'OTHER'
            fluid_type = 'not oil'

            if acc[1] == 1:
                is_nomal = 'NORMAL'
            if acc[0] ==  1:
                fluid_type = 'Oli'

            return { "meta": { "code": 200, "message": "success" }, "data": {
                'RFT': is_nomal, 'fluid_type': fluid_type,'mobility_score':
                acc[2] } }



    # parser = api.parser()
    # parser.add_argument('sdfsdfs',        help='ID of RFT tests', location='form')
    # @api.route('{}/predsssssssictions'.format(api_version))
    # class Predicdfsdfstions(Resource):
    #     @api.expect(parser)
    #     def post(self):
    #         args = parser.parse_args()
    #         # import ipdb; ipdb.set_trace()
    #         acc = mypd.predict(json.dumps(args))
    #         ans = 'OTHER'
    #         if acc[0] == 1:
    #             ans = 'NORMAL'
    #         return { "meta": { "code": 200, "message": "success" }, "data": { 'RFT': ans } }

    return app

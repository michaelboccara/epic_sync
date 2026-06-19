import argparse
import csv
import json
import logging
import math
import os
import urllib2
from datetime import datetime
from subprocess import check_call
from tempfile import gettempdir
from time import time, sleep, strptime, strftime

from google.cloud import storage
from google.cloud.exceptions import NotFound
import cv2
import numpy as np


class Epic:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        storage_client = storage.Client()
        self.bucket = storage_client.get_bucket(config['bucket'])

    def _read_file_from_mirror(self, key):
        try:
            blob = self.bucket.get_blob(key)
            return blob.download_as_string()
        except NotFound as e:
            logging.info(
                'error reading file from gs://{}/{}'.format(self.config['bucket'], key))
        return None

    def _read_file_from_url(self, url):
        for i in range(self.config['retries']):
            try:
                data = urllib2.urlopen(url)
                if data.code == 200:
                    return data.read()
            except BaseException:
                sleep(1)
        logging.info('error reading file from ' + url)
        return None

    def _read_json(self, data):
        return json.loads(data)

    def completed_dates(self):
        ret = []
        suffix = '.json'
        prefix = '{}/list/images_'.format(self.config['images_folder'])
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            ret.append(blob.name[len(prefix):-len(suffix)])
        return sorted(ret)

    def api_dates(self):
        ret = []
        url = self.config['api_url'] + '/all'
        data = self._read_file_from_url(url)
        dates_from_api = []
        for d in self._read_json(data):
            dates_from_api.append(d['date'])
        ret = dates_from_api
        return sorted(ret, reverse=True)

    def full_dates(self):
        return self.api_dates()

    def missing_dates(self):
        ret = []
        dates_from_api = self.api_dates()
        dates_from_mirror = self.completed_dates()
        missing_dates = set(dates_from_api) - set(dates_from_mirror)
        ret = missing_dates
        return sorted(ret, reverse=True)

    def edited_dates(self):
        ret = []
        dates_from_api = self.api_dates()
        dates_from_mirror = self.completed_dates()
        edited_dates = set([])
        common_dates = set(dates_from_api) & set(dates_from_mirror)
        for date in common_dates:
            logging.info('len for date: ' + date)
            num_images_api = len(self.image_list(date))
            num_images_archive = len(self.image_list_mirror(date))
            if num_images_api != num_images_archive:
                logging.info(
                    'At date: {}, api: {}, arch: {}'.format(
                        date, num_images_api, num_images_archive))
                edited_dates.add(date)
        ret = edited_dates
        return sorted(ret, reverse=True)

    def image_list(self, date):
        url = '{}/date/{}'.format(self.config['api_url'], date)
        data = self._read_file_from_url(url)
        return self._read_json(data)

    def image_list_mirror(self, date):
        bucket = self.config['bucket']
        key = '{}/list/images_{}.json'.format(
            self.config['images_folder'], date)
        data = self._read_file_from_mirror(key)
        return self._read_json(data)

    def _upload_file(self, path, key):
        if not self.args.dryrun:
            blob = storage.Blob(key, self.bucket)
            blob.upload_from_filename(path)

    def _upload_data(self, body, key, content_type):
        if not self.args.dryrun:
            blob = storage.Blob(key, self.bucket)
            blob.upload_from_string(body, content_type)

    def set_latest_date(self, date):
        source = '{}/list/images_{}.json'.format(
                self.config['images_folder'], date)
        key = self.config['latest_images_path']
        if not self.args.dryrun:
            blob = storage.Blob(source, self.bucket)
            self.bucket.copy_blob(blob, self.bucket, key)

    def get_date_from_image_name(self, image_name):
        date_part_from_name = image_name.split('_')[2]
        year = date_part_from_name[:4]
        month = date_part_from_name[4:6]
        day = date_part_from_name[6:8]
        return year, month, day

    def png(self, image_name):
        year, month, day = self.get_date_from_image_name(image_name)
        url = '{archive_url}/{year}/{month}/{day}/png/{image}.png'.format(
            archive_url=self.config['archive_url'],
            year=year,
            month=month,
            day=day,
            image=image_name)
        logging.info('Downloading ' + url)
        data = self._read_file_from_url(url)
        filename = os.path.join(gettempdir(), image_name + '.png')
        with open(filename, 'wb') as f:
            f.write(data)
        key = '{}/png/{}.png'.format(self.config['images_folder'], image_name)
        logging.info(
            'Uploading to gs://{}/{}'.format(self.config['bucket'], key))
        self._upload_file(filename, key)

    def jpgs(self, image_name):
        for res in self.config['res']:
            res_string = '{res}x{res}'.format(res=res)
            infile = os.path.join(gettempdir(), image_name + '.png')
            outfile = os.path.join(gettempdir(), image_name + '.jpg')
            cmd = 'convert {} -resize {} {}'.format(
                infile, res_string, outfile)
            check_call(cmd, shell=True)
            # compatibility hack
            if res == '2048':
                key = '{}/jpg/{}.jpg'.format(
                    self.config['images_folder'],
                    image_name)
            # NASA thumbnail
            elif res == '120':
                key = '{}/thumbs/{}.jpg'.format(
                    self.config['images_folder'],
                    image_name)
            else:
                key = '{}/jpg/{}/{}.jpg'.format(
                    self.config['images_folder'],
                    res,
                    image_name)
            logging.info(
                'Uploading to gs://{}/{}'.format(self.config['bucket'], key))
            self._upload_file(outfile, key)
            os.remove(outfile)

    def _boxPoints(self, ellipse):
        angle = ellipse[2]
        center = ellipse[0]
        size = ellipse[1]

        _angle = angle * np.pi / 180.
        b = np.cos(_angle) * 0.5
        a = np.sin(_angle) * 0.5

        v0 = (
            center[0] - a * size[1] - b * size[0],
            center[1] + b * size[1] - a * size[0])
        v1 = (
            center[0] + a * size[1] - b * size[0],
            center[1] - b * size[1] - a * size[0])
        v2 = (2 * center[0] - v0[0], 2 * center[1] - v0[1])
        v3 = (2 * center[0] - v1[0], 2 * center[1] - v1[1])

        vertices = [v0, v1, v2, v3]

        return vertices

    def _get_earth_contour(self, filename):
        im = cv2.imread(filename, 0)
        height, width = im.shape
        ret, thresh = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        idx = 0
        max_area = math.pi * (height / 2) ** 2
        for index, item in enumerate(contours):
            a = cv2.contourArea(item)
            if area < a < max_area:
                area = a
                idx = index
        cnt = contours[idx]
        return cnt

    def _create_debug_image(
            self,
            image_name,
            filename,
            contour,
            circle,
            ellipse):
        line_width = 1
        white = (255, 255, 255)
        blue = (255, 0, 0)
        green = (0, 255, 0)
        red = (0, 0, 255)
        im2 = cv2.imread(filename)
        center = circle[0]
        radius = circle[1]
        cv2.circle(
            im2,
            (int(center[0]),
             int(center[1])),
            int(radius),
            white,
            line_width)
        # main cross
        cv2.line(im2, (0, 2048 / 2), (2048, 2048 / 2), white, line_width)
        cv2.line(im2, (2048 / 2, 0), (2048 / 2, 2048), white, line_width)
        # ellipse cross
        box_points = self._boxPoints(ellipse)
        cv2.line(im2, (int(box_points[0][0]), int(box_points[0][1])), (
            int(box_points[2][0]), int(box_points[2][1])), blue, line_width)
        cv2.line(im2, (int(box_points[1][0]), int(box_points[1][1])), (
            int(box_points[3][0]), int(box_points[3][1])), blue, line_width)
        #
        cv2.ellipse(im2, ellipse, red, line_width)
        cv2.drawContours(im2, (contour), 0, green, line_width)
        debug_file = os.path.join(
            gettempdir(),
            '_debug_' + image_name + '.png')
        cv2.imwrite(debug_file, im2)
        key = '{}/debug/{}'.format(
            self.config['images_folder'],
            image_name + '.png')
        self._upload_file(debug_file, key)
        os.remove(debug_file)

    def _write_dimensions(self, circle, ellipse):
        center, radius = circle
        cx = center[0] / 2048
        cy = center[1] / 2048
        r = radius / 2048

        (ex, ey), (MA, ma), angle = ellipse
        ex_norm = ex / 2048
        ey_norm = ey / 2048
        e_width = MA / 2048
        e_height = ma / 2048

        dimensions = {
            'earth_circle': {
                'center': {'x': cx, 'y': cy},
                'radius': r
            },
            'earth_ellipse': {
                'center': {'x': ex_norm, 'y': ey_norm},
                'size': {'width': e_width, 'height': e_height},
                'angle': angle
            }
        }

        logging.info('dimensions: {}'.format(json.dumps(dimensions, indent=4)))
        return dimensions

    def bounding_shapes(self, image_name):
        filename = os.path.join(gettempdir(), image_name + '.png')
        contour = self._get_earth_contour(filename)
        circle = cv2.minEnclosingCircle(contour)
        ellipse = cv2.fitEllipse(contour)

        self._create_debug_image(
            image_name,
            filename,
            contour,
            circle,
            ellipse)

        dimensions = self._write_dimensions(circle, ellipse)
        cache = {
            'jpg': dimensions,
            'png': dimensions
        }

        return cache

    def check_ecllipse(self, coords):
        data = json.loads(coords)
        sun = [data['sun_j2000_position']['x'],
               data['sun_j2000_position']['y'],
               data['sun_j2000_position']['z']]
        sun_norm = np.divide(sun, np.linalg.norm(sun))
        lunar = [data['lunar_j2000_position']['x'],
                 data['lunar_j2000_position']['y'],
                 data['lunar_j2000_position']['z']]
        lunar_norm = np.divide(lunar, np.linalg.norm(lunar))
        dscovr = [data['dscovr_j2000_position']['x'],
                  data['dscovr_j2000_position']['y'],
                  data['dscovr_j2000_position']['z']]
        dscovr_norm = np.divide(dscovr, np.linalg.norm(dscovr))
        lunar_dscovr_cross = np.cross(lunar_norm, dscovr_norm)
        lunar_dscovr_norm = np.linalg.norm(lunar_dscovr_cross)
        lunar_sun_cross = np.cross(lunar_norm, sun_norm)
        lunar_sun_norm = np.linalg.norm(lunar_sun_cross)
        return lunar_dscovr_norm, lunar_sun_norm

    def run_dates(self, dates):
        #align = [['day', 'date', 'image', 'lunar dscovr', 'lunar sun', 'link']]
        for date in dates:
            logging.info('Working on date: ' + date)
            original_images_json = self.image_list(date)
            images_json = []
            logging.info(
                'Read json with {} images'.format(len(original_images_json)))
            for image in original_images_json:
                image_name = image['image']
                image_date = image['date']
                # fix image date
                try:
                    strptime(image_date, '%Y-%m-%d %H:%M:%S')
                except BaseException:
                    logging.error('Failed to parse image date for: ' + image_name)
                    continue
                try:
                    # fix json coming from the api
                    if isinstance(image['coords'], dict):
                        image['coords'] = json.dumps(
                            image['coords']).replace(
                            "'", '"').rstrip(',')
                    else:
                        image['coords'] = image[
                            'coords'].replace("'", '"').rstrip(',')
                    lunar_dscovr, lunar_sun = self.check_ecllipse(
                        image['coords'])
                    debug_url = 'https://storage.cloud.google.com/{}/{}/debug/{}'.format(
                        self.config['bucket'], self.config['images_folder'], image_name + '.png')
                    #align.append(
                    #    [date,
                    #     image['date'],
                    #     image_name,
                    #     lunar_dscovr,
                    #     lunar_sun,
                    #     debug_url])
                    logging.info('Working on image: ' + image_name)
                    self.png(image_name)
                    self.jpgs(image_name)
                    logging.debug('Extract bounding shapes for image: ' + image_name)
                    image['cache'] = self.bounding_shapes(image_name)
                    # delete png
                    logging.debug('Delete PNG for image: ' + image_name)
                    os.remove(os.path.join(gettempdir(), image_name + '.png'))
                except Exception as e:
                    logging.info(
                        'Skipped image: {} from date: {} because of an error: {}'.format(
                            image_name, date, e.message))
                    continue
                images_json.append(image)
            logging.info(
                'Uploading json with {} images'.format(
                    len(images_json)))
            self._upload_data(
                json.dumps(images_json, indent=4),
            '{}/list/images_{}.json'.format(
                self.config['images_folder'],
                date),
            'application/json')
            lists = self.completed_dates()
            self._upload_data(
                json.dumps(lists, indent=4),
                self.config['available_dates_path'],
                'application/json')
            if self.args.dates is None:
                self.set_latest_date(lists[-1])
        #filename = os.path.join(
        #    gettempdir(),
        #    datetime.now().strftime('%s') + '.csv')
        #with open(filename, 'wb') as f:
        #    csv_writer = csv.writer(f)
        #    csv_writer.writerows(align)

    def run(self):
        # dates = ['2016-07-05', '2016-03-09', '2017-02-12'] # moon in frame,
        # lunar eclipse, none
        #self.run_dates(dates)
        if self.args.dates is None:
            if self.args.full:
                dates = self.full_dates()
            else: 
                dates = self.missing_dates()
                self.run_dates(dates)
                dates = []
                if not self.args.nooverwrite:
                    dates = self.edited_dates()
                    self.run_dates(dates)
        else:
            dates = self.args.dates.split(',')
            for d in dates:
                try:
                    strptime(d.strip(), '%Y-%m-%d')
                except BaseException:
                    logging.error('"{}" not a valid date (%Y-%m-%d)'.format(d))
                    exit(-1)
                self.run_dates(dates)


def main():
    def _parse_arguments():
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--full',
            help='Full sync',
            action='store_true')
        parser.add_argument(
            '--nooverwrite',
            help='Not overwriting existing dates',
            action='store_true')
        parser.add_argument(
            '--dryrun',
            help='Not writing to mirror',
            action='store_true')
        parser.add_argument(
            '--verbose',
            help='Print debug messages',
            action='store_true')
        parser.add_argument(
            '--dev',
            help='Use dev bucket',
            action='store_true')
        parser.add_argument(
            '--enhanced',
            help='Sync enhanced images',
            action='store_true')
        group.add_argument(
            '--dates',
            help='''
                 Comma separated list of dates to sync.
                 Example: "2016-07-05, 2016-03-09".
                 Implies not syncing lates date.
                 ''',
            default=None)
        return parser.parse_args()

    def _config(args):
        if args.verbose:
            logging.basicConfig(level=logging.INFO)

        config = {}

        if args.dev:
            config['bucket'] = 'blueturn-content-dev'
        else:
            config['bucket'] = 'content.blueturn.earth'

        base_url = 'http://epic.gsfc.nasa.gov'
        if args.enhanced:
            config['api_url'] = base_url + '/api/enhanced'
            config['archive_url'] = base_url + '/archive/enhanced'
            config['images_folder'] = 'enhanced_images'
            config[
                'available_dates_path'] = 'enhanced_images/available_dates.json'
            config['latest_images_path'] = 'enhanced_images/images_latest.json'
        else:
            config['api_url'] = base_url + '/api/natural'
            config['archive_url'] = base_url + '/archive/natural'
            config['images_folder'] = 'images'
            config['available_dates_path'] = 'images/available_dates.json'
            config['latest_images_path'] = 'images/images_latest.json'

        config['retries'] = 5
        config['res'] = ['2048', '1024', '512', '256', '120']
        return config

    args = _parse_arguments()
    config = _config(args)

    logging.info('run Epic');
    epic = Epic(args, config)
    epic.run()


if __name__ == '__main__':
    main()

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


db = SQLAlchemy()


class CommonMixin(object):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(25))
    description = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)


class Organization(CommonMixin, db.Model):
    pass


class Role(CommonMixin, db.Model):
    pass


class Resource(CommonMixin, db.Model):
    resource = db.Column(db.String(128))


class Client(CommonMixin, db.Model):
    resource = db.Column(db.Integer, db.ForeignKey("resource.id"), nullable=False)
    organization = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)
    approval_state = db.Column(db.Integer, default=0)


class User(CommonMixin, db.Model):
    email = db.Column(db.String(128))
    hashed_pw = db.Column(db.String(128))
    role = db.Column(db.Integer, db.ForeignKey("role.id"), nullable=False)
    organization = db.Column(db.Integer, db.ForeignKey("organization.id"), nullable=False)

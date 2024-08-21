/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENDEC_H
#define ENDEC_H

#include "gmp.h"

class Endec {
  private:
    bool debug_ = false;
    double precision_;

  public:
    Endec(double p, bool debug = false): debug_(debug), precision_(p) {}

    void encode(mpz_t& result, const double& number) {
      int64_t temp = static_cast<int64_t>(number * precision_);
      uint64_t output_number = static_cast<uint64_t>(temp);

      mpz_set_ui(result, output_number);
      if (debug_) printf("Encoding using (p %f): input %f, output %lu\n", precision_, number, output_number);

    }

    double decode(const mpz_t& number) {
      uint64_t output_num = mpz_get_ui(number);
      int64_t sint = static_cast<int64_t>(output_num);
      double result = sint / precision_;

      if (debug_) gmp_printf("Decoding using (p %f): input %Zd, output %f\n", precision_, number, result);
      return result;
    }
};

#endif // ENDEC_H

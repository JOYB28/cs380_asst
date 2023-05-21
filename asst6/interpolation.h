#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include "rigtform.h"
#include "quat.h"

namespace Interpolation {

    // Linear interpolation of two coordinate vectors
    inline Cvec3 lerp(const Cvec3& c0, const Cvec3& c1, const double& alpha) {
      double resultX = (1 - alpha) * c0[0] + alpha * c1[0];
      double resultY = (1 - alpha) * c0[1] + alpha * c1[1];
      double resultZ = (1 - alpha) * c0[2] + alpha * c1[2];
      std::cout << "lerp! alpha: " << alpha << std::endl;
      std::cout << "c0.x: " << c0[0] << ", c0.y: " << c0[1] << ", c0.z: " << c0[2] << std::endl;
      std::cout << "c1.x: " << c1[0] << ", c1.y: " << c1[1] << ", c1.z: " << c1[2] << std::endl;
      std::cout << "resultX: " << resultX << ", resultY: " << resultY << ", resultZ: " << resultZ << std::endl;
      return Cvec3(resultX, resultY, resultZ);
    }

    // Spherical interpolation of two quaternions
    inline Quat slerp(const Quat& q0, const Quat& q1, const double& alpha) {
      std::cout << "slerp! alpha: " << alpha << std::endl;
      std::cout << "q0.w: " << q0[0] << ", q0.x: " << q0[1] << ", q0.y: " << q0[2] << ", q0.z: " << q0[3] << std::endl;
      std::cout << "q1.w: " << q1[0] << ", q1.x: " << q1[1] << ", q1.y: " << q1[2] << ", q1.z: " << q1[3] << std::endl;

      // calculate quatAngle
      double cosValue = dot(q0,q1);

      double sinValue = norm2(q0 * q1);
//    double sinValue = norm(cross(Cvec3(q0[1], q0[2], q0[3]), Cvec3(q1[1], q1[2], q1[3])));

      double quatAngle = atan2(sinValue, cosValue);

      Quat result;
      std::cout << "quatAngle: " << quatAngle << ", cosValue: " << cosValue << ", sinValue: " << sinValue << std::endl;
      if (quatAngle == 0) {
        std::cout << "quatAngle is 0. Use q0 instead" << quatAngle << std::endl;
        result = q0;
      } else {
        result = q0.operator*(sin((1 - alpha) * quatAngle) / sin(quatAngle)) + q1.operator*(sin(alpha *quatAngle) / sin(quatAngle));
      }

      std::cout << "result.w: " << result[0] << ", result.x: " << result[1] << ", result.y: " << result[2] << ", result.z: " << result[3] << std::endl;
      return result;
    }

    // Linear interpolation of two RigTForms
    // Note: you should use the lerp and slerp functions you implemented above
    inline RigTForm Linear(const RigTForm& rbt0, const RigTForm& rbt1, const double& alpha) {
      Cvec3 t = lerp(rbt0.getTranslation(), rbt1.getTranslation(), alpha);
      Quat r = slerp(rbt0.getRotation(), rbt1.getRotation(), alpha);
      return RigTForm(t, r);
    }

    inline Cvec3 catmullRom(Cvec3 c_1, Cvec3 c0, Cvec3 c1, Cvec3 c2, double alpha) {
      Cvec3 d = (c1 - c_1) * (1.0 / 6.0) + c0;
      Cvec3 e = (c0 - c2) * (1.0 / 6.0) + c1;

      Cvec3 p01 = lerp(c0, d, alpha);
      Cvec3 p12 = lerp(d, e, alpha);
      Cvec3 p23 = lerp(e, c1, alpha);
      Cvec3 p012 = lerp(p01, p12, alpha);
      Cvec3 p123 = lerp(p12, p23, alpha);
      Cvec3 p = lerp(p012, p123, alpha);

      return p;
    }

    inline Quat controlPoint(Quat q_1, Quat q0, Quat q1, double sign) {
      std::cout << "[controlPoint] start!" << std::endl;
      std::cout << "q_1.w: " << q_1[0] << ", q_1.x: " << q_1[1] << ", q_1.y: " << q_1[2] << ", q_1.z: " << q_1[3] << std::endl;
      std::cout << "q0.w: " << q0[0] << ", q0.x: " << q0[1] << ", q0.y: " << q0[2] << ", q0.z: " << q0[3] << std::endl;
      std::cout << "q1.w: " << q1[0] << ", q1.x: " << q1[1] << ", q1.y: " << q1[2] << ", q1.z: " << q1[3] << std::endl;

      // 같은 경우
      if (q_1[0] == q1[0]
        && q_1[1] == q1[1]
        && q_1[2] == q1[2]
        && q_1[3] == q1[3]
      ) {
        std::cout << "[controlPoint] identity!" << std::endl;
        return q0;
      }
      Quat base = q1 * inv(q_1);

      // conditionally negate
      if (base[0] < 0) {
        std::cout << "[controlPoint] conditionally negate!" << std::endl;
        base = base * -1;
      }

      std::cout << "[controlPoint] !!" << std::endl;
      return pow(base, 1.0 / 6.0 * sign) * q0;
    }

    inline Quat catmullRom(Quat q_1, Quat q0, Quat q1, Quat q2, double alpha) {
      // d = (q1 * inv(q-1)) ^ 1/6 * q0
      // e = (q2 * inv(q0)) ^ -1/6 * q1
      Quat d = controlPoint(q_1, q0, q1, 1.0);
      Quat e = controlPoint(q0, q1, q2, -1.0);

      Quat p01 = slerp(q0, d, alpha);
      Quat p12 = slerp(d, e, alpha);
      Quat p23 = slerp(e, q1, alpha);
      Quat p012 = slerp(p01, p12, alpha);
      Quat p123 = slerp(p12, p23, alpha);
      Quat p = slerp(p012, p123, alpha);

      return p;
    }

    // Catmull-Rom interpolation of two RigTForms
    // Note: To Catmull-Rom interpolate two RigTFrom rbt0 and rbt1, we need 4 Keyframes keyframe rbt_1, rbt0, rbt1, rbt2.
    // 		 You can use the lerp and slerp functions you implemented above.
    inline RigTForm CatmullRom(const RigTForm& rbt_1, const RigTForm& rbt0, const RigTForm& rbt1, const RigTForm& rbt2, const double& alpha) {
      Cvec3 t = catmullRom(
        rbt_1.getTranslation(),
        rbt0.getTranslation(),
        rbt1.getTranslation(),
        rbt2.getTranslation(),
        alpha
      );

      Quat r = catmullRom(
        rbt_1.getRotation(),
        rbt0.getRotation(),
        rbt1.getRotation(),
        rbt2.getRotation(),
        alpha
      );

      return RigTForm(t, r);
    }

}
#endif
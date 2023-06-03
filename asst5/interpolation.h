#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include "rigtform.h"

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
//    Quat base = q1 * inv(q0);
//    // conditionally negate
//    if (base[0] < 0) {
//      base = base * -1;
//    }

    std::cout << "slerp! alpha: " << alpha << std::endl;
    std::cout << "q0.w: " << q0[0] << ", q0.x: " << q0[1] << ", q0.y: " << q0[2] << ", q0.z: " << q0[3] << std::endl;
    std::cout << "q1.w: " << q1[0] << ", q1.x: " << q1[1] << ", q1.y: " << q1[2] << ", q1.z: " << q1[3] << std::endl;

    // calculate quatAngle
    double cosValue = dot(q0,q1);

    // 혹은 sin(x)^2 + cos(x)^2 = 1 을 이용해서 해도 될듯
    // 내가 했던 sinValue2는 어떻게 나온건지 기억이 안남..
    double sinValue2 = norm2(q0 * q1);
    // 차라리 아래처럼 sin = (q0 x q1) / |q0 x q1| 으로 하면 되는데 과제할때 이게 안돼서 여러가지 해보다가 된듯..
    double sinValue = norm(cross(Cvec3(q0[1], q0[2], q0[3]), Cvec3(q1[1], q1[2], q1[3])));

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
		return RigTForm(t, r);	// Replace this value with your own code
	}

}
#endif
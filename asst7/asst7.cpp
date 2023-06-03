////////////////////////////////////////////////////////////////////////
//
//   KAIST, Spring 2023
//   CS380: Introduction to Computer Graphics
//   Instructor: Minhyuk Sung (mhsung@kaist.ac.kr)
//   Last Update: Juil Koo (63days@kaist.ac.kr)
//
////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
// If your OS is LINUX, uncomment the line below.
//#include <tr1/memory>

#include <GL/glew.h>

#ifdef __MAC__

#   include <GLUT/glut.h>

#else
#   include <GL/glut.h>
#endif

#include "cvec.h"
#include "matrix4.h"
#include "rigtform.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"
#include "arcball.h"
#include "asstcommon.h"
#include "scenegraph.h"
#include "drawer.h"
#include "picker.h"
#include "geometry.h"

using namespace std;      // for string, vector, iostream, and other standard C++ stuff
// If your OS is LINUX, uncomment the line below.
//using namespace tr1; // for shared_ptr

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.3.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.3. Make sure that your machine supports the version of GLSL you
// are using. In particular, on Mac OS X currently there is no way of using
// OpenGL 3.x with GLSL 1.3 when GLUT is used.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
const bool g_Gl2Compatible = true;

static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length

static int g_windowWidth = 512; // TODO: 512
static int g_windowHeight = 512; // TODO: 512
static bool g_mouseClickDown = false;    // is the mouse button pressed
// 왼쪽, 오른쪽, 스크롤(middle) 클릭
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;
static bool g_picking = false;

// --------- Materials
static shared_ptr<Material> g_redDiffuseMat,
  g_blueDiffuseMat,
  g_bumpFloorMat,
  g_arcballMat,
  g_pickingMat,
  g_lightMat;

shared_ptr<Material> g_overridingMaterial;

// --------- Geometry
typedef SgGeometryShapeNode MyShapeNode;

// ===================================================================
// Declare the scene graph and pointers to suitable nodes in the scene
// graph
// ===================================================================

static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode; // used later when you do picking
static shared_ptr<SgRbtNode> g_nullRbtNode = shared_ptr<SgRbtNode>();

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_cube2, g_sphere;

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space
//static RigTForm g_skyRbt = RigTForm(Cvec3(0.0, 0.25, 4.0));
//static RigTForm g_objectRbt[2] = {
//  RigTForm(Cvec3(0.75, 0, 0)),
//  RigTForm(Cvec3(-0.75, 0, 0)),
//};
//static Cvec3f g_objectColors[2] = {
//  Cvec3f(1, 0, 0),
//  Cvec3f(0, 0, 1),
//};
static Cvec3f g_arcballColor = Cvec3f(0, 1, 0);

//static RigTForm g_currentObjectRbt = g_objectRbt[0];

static RigTForm g_currentEyeRbt;
// 0: sky, 1: robot1, 2: robot2
static int g_currentEyeIdx = 0;
static string g_eyeNames[3] = {"sky", "robot1", "robot2"};

static int g_manipulatedObject[3] = {
  -1,
  0,
  1,
};

static int g_currentSkyFrame = 1; // 0: world-sky, 1: sky-sky
static string g_skyFrameNames[2] = {"world-sky", "sky-sky"};

static double g_arcballScreenRadius = 0.25 * min(g_windowWidth, g_windowHeight);
static double g_arcballScale = 0.01;

///////////////// END OF G L O B A L S //////////////////////////////////////////////////


static void initGround() {
  int ibLen, vbLen;
  getPlaneVbIbLen(vbLen, ibLen);

  // Temporary storage for cube Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makePlane(g_groundSize*2, vtx.begin(), idx.begin());
  g_ground.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);

  // Temporary storage for cube Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeCube(1, vtx.begin(), idx.begin());
  g_cube.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere() {
  int ibLen, vbLen;
  getSphereVbIbLen(20, 10, vbLen, ibLen);

  // Temporary storage for sphere Geometry
  vector<VertexPNTBX> vtx(vbLen);
  vector<unsigned short> idx(ibLen);
  makeSphere(1, 20, 10, vtx.begin(), idx.begin());
  g_sphere.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vtx.size(), idx.size()));
}

// takes a projection matrix and send to the the shaders
inline void sendProjectionMatrix(Uniforms& uniforms, const Matrix4& projMatrix) {
  uniforms.put("uProjMatrix", projMatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS380_PI / 180;
    g_frustFovY =
      atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) /
      RAD_PER_DEG;
  }
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
    g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
    g_frustNear, g_frustFar);
}

static void drawStuff(bool picking, string context) {
  cout << "drawStuff (context: " << context << " , picking: " << picking << ")" << endl;

  // Declare an empty uniforms
  Uniforms uniforms;

  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  // send proj. matrix to be stored by uniforms,
  // as opposed to the current vtx shader
  sendProjectionMatrix(uniforms, projmat);

  // use the skyRbt as the eyeRbt (스카이캠) Rbt body transformation
  // -- 과제하기 위해 eye frame을 바꿔야함
  if (g_currentEyeIdx == 0) {
    cout << "getPathAccumRbt debug 1" << endl;
    g_currentEyeRbt = getPathAccumRbt(g_world, g_skyNode);
  } else if (g_currentEyeIdx == 1) {
    cout << "getPathAccumRbt debug 2" << endl;
    g_currentEyeRbt = getPathAccumRbt(g_world, g_robot1Node);
  } else {
    cout << "getPathAccumRbt debug 3" << endl;
    g_currentEyeRbt = getPathAccumRbt(g_world, g_robot2Node);
  }

  const RigTForm eyeRbt = g_currentEyeRbt;
//  cout << "eyeRbt x: " << eyeRbt.getTranslation()[0] << ", y: " << eyeRbt.getTranslation()[1] << ", z: " << eyeRbt.getTranslation()[2] << endl;
  const RigTForm invEyeRbt = inv(eyeRbt);

  const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1)); // g_light1 position in eye coordinates
  const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1)); // g_light2 position in eye coordinates
  // send the eye space coordinates of lights to uniforms
  uniforms.put("uLight", eyeLight1);
  uniforms.put("uLight2", eyeLight2);

  if (!picking) {
    // initialize the drawer with our uniforms, as opposed to curSS
    Drawer drawer(invEyeRbt, uniforms);

    g_world->accept(drawer);

    // draw arcball
    // ============
    cout << "g_currentPickedRbtNode: " << g_currentPickedRbtNode << endl;
    cout << "g_nullRbtNode: " << g_nullRbtNode << endl;
    cout << "g_currentPickedRbtNode == g_nullRbtNode 결과: " << (g_currentPickedRbtNode == g_nullRbtNode) << endl;
    string currentEyeName = g_eyeNames[g_currentEyeIdx];
    string currentSkyFrame = g_skyFrameNames[g_currentSkyFrame];
    if (g_currentPickedRbtNode != g_nullRbtNode || (currentEyeName == "sky" && currentSkyFrame == "world-sky")) {
      cout << "getPathAccumRbt debug 4" << endl;
      RigTForm arcballPlace;
      if (currentEyeName == "sky" && currentSkyFrame == "world-sky") {
        // world-sky 프레임이라고 가정할 수 있음
        arcballPlace = RigTForm(Cvec3(0, 0, 0));
      } else {
        arcballPlace = getPathAccumRbt(g_world, g_currentPickedRbtNode);
      }

      double z;
      if (currentEyeName == "sky" && currentSkyFrame == "world-sky") {
        // world-sky 프레임이라고 가정할 수 있음
        z = (invEyeRbt * RigTForm(Cvec3(0, 0, 0))).getTranslation()[2];
      } else {
        z = (invEyeRbt * arcballPlace).getTranslation()[2];
      }

      // 눌려있는 동안 업데이트 X
      if (!(g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton))) {
        g_arcballScale = getScreenToEyeScale(z, g_frustFovY, g_windowHeight);
      }

      double radius = g_arcballScale * g_arcballScreenRadius;
      Matrix4 scale = Matrix4::makeScale(Cvec3(1, 1, 1) * radius);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      Matrix4 MVM;
      if (currentEyeName == "sky" && currentSkyFrame == "world-sky") {
        // world-sky 프레임이라고 가정할 수 있음
        MVM = rigTFormToMatrix(invEyeRbt * RigTForm(Cvec3(0, 0, 0))) * scale;
      } else {
        MVM = rigTFormToMatrix(invEyeRbt * arcballPlace) * scale;
      }

      sendModelViewNormalMatrix(uniforms, MVM, normalMatrix(MVM));
      g_arcballMat->draw(*g_sphere, uniforms);
    }
  }
  else {
    // intialize the picker with our uniforms, as opposed to curSS
    Picker picker(invEyeRbt, uniforms);

    // set overiding material to our picking material
    g_overridingMaterial = g_pickingMat;

    g_world->accept(picker);

    // unset the overriding material
    g_overridingMaterial.reset();

    glFlush();
    g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX, g_mouseClickY);
    if (g_currentPickedRbtNode == g_groundNode)
//      g_currentPickedRbtNode = g_skyNode;
      g_currentPickedRbtNode = g_nullRbtNode;   // set to NULL
  }
}

static void display() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

  drawStuff(g_picking, "display()");

  if (!g_picking) {
    // show the back buffer (where we rendered stuff)
    glutSwapBuffers();
  }

  checkGlErrors();
}

static void pick() {
  // We need to set the clear color to black, for pick rendering.
  // so let's save the clear color
  GLdouble clearColor[4];
  glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);

  glClearColor(0, 0, 0, 0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  drawStuff(true, "pick()");

  // Uncomment below and comment out the glutPostRedisplay in mouse(...) call back
  // to see result of the pick rendering pass
//   glutSwapBuffers();

  //Now set back the clear color
  glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);

  checkGlErrors();
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  g_arcballScreenRadius = 0.25 * min(g_windowWidth, g_windowHeight);
  glViewport(0, 0, w, h);
  cerr << "Size of window is now " << w << "x" << h << endl;
  cerr << "Arcball Screen Radius is now " << g_arcballScreenRadius <<  endl;
  updateFrustFovY();
  glutPostRedisplay();
}

Cvec3 getArcballPoint(Cvec2 screenPoint, Cvec3 center, double radius, Cvec3 eye) {
  cout << "getArcballPoint!!" << endl;

  Cvec3 ray = getModelViewRay(screenPoint, g_frustFovY, g_windowWidth, g_windowHeight);
  cout << "ray vector before x: " << ray[0] << ", y: " << ray[1] << ", z: " << ray[2] << endl;
  ray = Cvec3(-ray[0], -ray[1], -ray[2]);
  cout << "ray vector after x: " << ray[0] << ", y: " << ray[1] << ", z: " << ray[2] << endl;

  cout << "screenPoint x: " << screenPoint[0] << ", y: " << screenPoint[1] << endl;
  Cvec3 centerRay = center - eye;
  cout << "radius: " << radius << endl;

  cout << "eye point x: " << eye[0] << ", y: " << eye[1] << ", z: " << eye[2] << endl;
  cout << "center point x: " << center[0] << ", y: " << center[1] << ", z: " << center[2] << endl;
  cout << "eye->center vector x: " << centerRay[0] << ", y: " << centerRay[1] << ", z: " << centerRay[2] << endl;

  double a = norm2(ray);
  cout << "a: " << a << endl;
  double b = -2 * dot(ray, centerRay);
  cout << "b: " << b << endl;
  double c = norm2(centerRay) - radius * radius;
  cout << "c: " << c << endl;
  double d = b * b - 4 * a * c;
  cout << "b^2 - 4ac: " << d << endl;
  bool intersect = b * b - 4 * a * c >= 0;
  cout << "intersect: " << intersect << endl;

  double root;
  if (intersect) {
    root = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
    cout << "root1: " << root << endl;
  } else {
    root = -b / (2 * a);
    cout << "not intersect! root2" << endl;
  }
  Cvec3 point = eye + ray * root;
  double distanceOfCenter = sqrt((point[0] - center[0]) * (point[0] - center[0]) + (point[1] - center[1]) * (point[1] - center[1]) + (point[2] - center[2]) * (point[2] - center[2]));
  cout << "distanceOfCenter: " << distanceOfCenter << endl;
  if (abs(distanceOfCenter - radius) > CS380_EPS) {
    point = center + (point - center) * radius / distanceOfCenter;
    distanceOfCenter = sqrt((point[0] - center[0]) * (point[0] - center[0]) + (point[1] - center[1]) * (point[1] - center[1]) + (point[2] - center[2]) * (point[2] - center[2]));
    cout << "new distanceOfCenter: " << distanceOfCenter << endl;
  }
  cout << "point x: " << point[0] << ", y: " << point[1] << ", z: " << point[2] << endl;
  return point;
}

static void motion(const int x, const int y) {
  cout << "motion! x:" << x << " y:" << y << endl;

  const double dx = x - g_mouseClickX;
  const double dy = g_windowHeight - y - 1 - g_mouseClickY;
  cout << "dx: " << dx << ", dy: " << dy << endl;

  RigTForm auxFrame;
  if (g_currentEyeIdx == 0) {
    g_currentEyeRbt = g_skyNode->getRbt();
  } else if (g_currentEyeIdx == 1) {
    cout << "getPathAccumRbt debug 5" << endl;
    g_currentEyeRbt = getPathAccumRbt(g_world, g_robot1Node);
  } else {
    cout << "getPathAccumRbt debug 6" << endl;
    g_currentEyeRbt = getPathAccumRbt(g_world, g_robot2Node);
  }

  bool arcballRotation = false;
  RigTForm arcballRotationRbt = RigTForm();
  // arcball 회전
  string currentEyeName = g_eyeNames[g_currentEyeIdx];
  string currentSkyFrame = g_skyFrameNames[g_currentSkyFrame];
  if (g_currentPickedRbtNode != g_nullRbtNode || (currentEyeName == "sky" && currentSkyFrame == "world-sky")) {

    Cvec3 centerOfArcball;
    if (g_currentPickedRbtNode == g_nullRbtNode) {
      // world-sky 프레임이라고 가정할 수 있음
      centerOfArcball = Cvec3(0, 0, 0);
    } else {
      // object
      cout << "getPathAccumRbt debug 7" << endl;
      centerOfArcball = getPathAccumRbt(g_world, g_currentPickedRbtNode).getTranslation();
    }

    double radius = g_arcballScale * g_arcballScreenRadius;
    cout << "radius: " << radius << endl;
    cout << "scale: " << g_arcballScale << endl;
    cout << "screenRadius: " << g_arcballScreenRadius << endl;
    Cvec3 eye = g_currentEyeRbt.getTranslation();
    Cvec3 arcballPoint0 = getArcballPoint(Cvec2(g_mouseClickX, g_mouseClickY), centerOfArcball, radius, eye);
    cout << "arcballPoint0 x: " << arcballPoint0[0] << ", y: " << arcballPoint0[1] << ", z: " << arcballPoint0[2] << endl;
    double distanceOfCenter0 = sqrt((arcballPoint0[0] - centerOfArcball[0]) * (arcballPoint0[0] - centerOfArcball[0]) + (arcballPoint0[1] - centerOfArcball[1]) * (arcballPoint0[1] - centerOfArcball[1]) + (arcballPoint0[2] - centerOfArcball[2]) * (arcballPoint0[2] - centerOfArcball[2]));
    cout << "distanceOfCenter0: " << distanceOfCenter0 << endl;

    Cvec3 arcballPoint1 = getArcballPoint(Cvec2(g_mouseClickX + dx, g_mouseClickY + dy), centerOfArcball, radius, eye);;
    cout << "arcballPoint1 x: " << arcballPoint1[0] << ", y: " << arcballPoint1[1] << ", z: " << arcballPoint1[2] << endl;

    double distanceOfCenter1 = sqrt((arcballPoint1[0] - centerOfArcball[0]) * (arcballPoint1[0] - centerOfArcball[0]) + (arcballPoint1[1] - centerOfArcball[1]) * (arcballPoint1[1] - centerOfArcball[1]) + (arcballPoint1[2] - centerOfArcball[2]) * (arcballPoint1[2] - centerOfArcball[2]));
    cout << "distanceOfCenter1: " << distanceOfCenter1 << endl;

    Cvec3 arcballVector0 = normalize(arcballPoint0 - centerOfArcball);
    Cvec3 arcballVector1 = normalize(arcballPoint1 - centerOfArcball);
    Quat quatVector0 = Quat(0, arcballVector0);
//    cout << "quatVector0 i: " << arcballVector0[1] << ", j: " << arcballVector0[2] << ", k: " << arcballVector0[3] << ", w: " << arcballVector0[0] << endl;
    Quat quatVector1 = Quat(0, arcballVector1);
//    cout << "quatVector1 i: " << arcballVector1[1] << ", j: " << arcballVector1[2] << ", k: " << arcballVector1[3] << ", w: " << arcballVector1[0] << endl;

//    cout << "arcballRotation: true" << endl;
    arcballRotation = true;
//    if (manipulatedObjectName == "sky")
    if (g_currentPickedRbtNode == g_nullRbtNode) {
      // world-sky 프레임이라고 가정할 수 있음
      arcballRotationRbt.setRotation(quatVector1 * (quatVector0 * -1));
      arcballRotationRbt = inv(arcballRotationRbt);
    } else {
      // cube
      arcballRotationRbt.setRotation(quatVector1 * (quatVector0 * -1));
    }
  }

  cout << "arcballRotation: " << arcballRotation << endl;

  if (g_currentPickedRbtNode == g_nullRbtNode) {
    // object가 sky
    cout << "object is sky!!" << endl;
    if (g_currentEyeIdx != 0) {
      cout << "debug1" << endl;
      // eye frame이 sky가 아닌 경우
      return;
    }
    else if (g_skyFrameNames[g_currentSkyFrame] == "world-sky") {
      cout << "debug2" << endl;
      // world-sky
      cout << "eye is world-sky!!" << endl;
      auxFrame = linFact(g_skyNode->getRbt());
    } else {
      cout << "debug3" << endl;
      // sky-sky
      cout << "eys is sky-sky!!" << endl;
      auxFrame = transFact(g_skyNode->getRbt()) * linFact(g_skyNode->getRbt());
    }
  }
  else {
    cout << "debug4" << endl;
    // object가 sky가 아님 (로봇 등)
    cout << "object is not sky!!" << endl;
    // eye가 sky
    if (g_currentEyeIdx == 0) {
      cout << "getPathAccumRbt debug 8" << endl;
      auxFrame = inv(getPathAccumRbt(g_world, g_currentPickedRbtNode, 1))
        * transFact(getPathAccumRbt(g_world, g_currentPickedRbtNode))
        * linFact(getPathAccumRbt(g_world, g_skyNode));
    }
    // eye가 다른 robot
    else {
      cout << "getPathAccumRbt debug 9" << endl;
      auxFrame = inv(getPathAccumRbt(g_world, g_currentPickedRbtNode, 1))
        * getPathAccumRbt(g_world, g_currentPickedRbtNode);
    }
  }

  RigTForm m;
  if (g_mouseLClickButton && !g_mouseRClickButton) { // left button down?
    cout << "debut AA" << endl;
    if (arcballRotation) {
      cout << "debut A1" << endl;
      m = arcballRotationRbt;
    } else {
      cout << "debut A2" << endl;
      // 둘중에 하나임
      m = RigTForm(Quat::makeXRotation(dy) * Quat::makeYRotation(-dx));
    }
  } else if (g_mouseRClickButton && !g_mouseLClickButton) { // right button down?
    cout << "debut CC" << endl;
//    cout << "g_arcballScale1: " << g_arcballScale << endl;
    // world-sky이면서, object, eye 둘다 sky
    if (currentEyeName == "sky" && currentSkyFrame == "world-sky") {
      cout << "debut C1" << endl;
//    if (g_eyeNames[g_currentEyeIdx] == "sky"
//      && g_manipulatedObjectNames[g_currentManipulatedObjectIdx] == "sky"
//      && g_skyFrameNames[g_currentSkyFrame] == "world-sky"
//    ) {
      m = RigTForm(Cvec3(-dx, -dy, 0) * g_arcballScale);
    } else {
      cout << "debut C2" << endl;
      m = RigTForm(Cvec3(dx, dy, 0) * g_arcballScale);
    }
  } else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton)) {  // middle or (left and right) button down?
    cout << "debut B" << endl;
//    cout << "g_arcballScale2: " << g_arcballScale << endl;
//    cout << "debugxxxxxx" << endl;
    m = RigTForm(Cvec3(0, 0, -dy) * g_arcballScale);
//    g_arcballScreenRadius = (g_arcballScale / 0.01);
  }

  cout << "debut D" << endl;
  // AMA^-1
//  cout << "debug10" << endl;
  RigTForm ama = auxFrame * m * inv(auxFrame);
//  cout << "debug11" << endl;

  cout << "debut E" << endl;
  if (g_mouseClickDown) {
    cout << "debut F" << endl;
    if (g_currentPickedRbtNode != g_nullRbtNode) {
      cout << "debut F1" << endl;
      cout << "hello" << endl;
      g_currentPickedRbtNode->setRbt(ama * g_currentPickedRbtNode->getRbt());
    } else {
      cout << "debug F2" << endl;
      g_skyNode->setRbt(ama * g_skyNode->getRbt());
    }

    cout << "debut F3" << endl;
    glutPostRedisplay(); // we always redraw if we changed the scene
  }
  cout << "debut G" << endl;

  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;
}

static void mouse(const int button, const int state, const int x, const int y) {
  cout << "mouse! button:" << button << " state:" << state << " x:" << x << " y:" << y << endl;
  g_mouseClickX = x;
  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system
  g_mouseClickY = g_windowHeight - y - 1;

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

  if (g_picking && g_mouseLClickButton) {
    pick();
    g_picking = false;
    cout << "picking end! g_picking: true -> false" << endl;
  }

  // 마우스 떼졌을때 redraw
  // TODO: 다시 uncomment
  glutPostRedisplay();
}

static void keyboard(const unsigned char key, const int x, const int y) {
  switch (key) {
    case 27:
      exit(0);                                  // ESC
    case 'h':
      cout << " ============== H E L P ==============\n\n"
           << "h\t\thelp menu\n"
           << "s\t\tsave screenshot\n"
           << "f\t\tToggle flat shading on/off.\n"
           << "o\t\tCycle object to edit\n"
           << "v\t\tCycle view\n"
           << "drag left mouse to rotate\n" << endl;
      break;
    case 's':
      glFlush();
      writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
      break;
    case 'v':
      g_currentEyeIdx = (g_currentEyeIdx + 1) % 3;
      cout << "현재 eye: " << g_eyeNames[g_currentEyeIdx] << endl;

      if (g_currentEyeIdx == 1) {
        cout << "eye가 robot이라서 pickedRbtNode를 robot1로 업데이트!" << endl;
        g_currentPickedRbtNode = g_robot1Node;
      } else if (g_currentEyeIdx == 2) {
        cout << "eye가 robot이라서 pickedRbtNode를 robot2로 업데이트!" << endl;
        g_currentPickedRbtNode = g_robot2Node;
      } else {
        // eye가 sky가 될때 초기화
        g_currentPickedRbtNode = g_nullRbtNode;
      }
      cout << "pickedRbt 초기화!" << endl;
      break;
//    case 'o':
//      g_currentManipulatedObjectIdx = (g_currentManipulatedObjectIdx + 1) % 3;
//      cout << "현재 object: " << g_manipulatedObjectNames[g_currentManipulatedObjectIdx] << endl;
//      break;
    case 'm':
      if (g_currentPickedRbtNode == g_nullRbtNode
        && g_eyeNames[g_currentEyeIdx] == "sky"
      ) {
        g_currentSkyFrame = (g_currentSkyFrame + 1) % 2;
        cout << "현재 sky frame: " << g_skyFrameNames[g_currentSkyFrame] << endl;
      } else {
        cout << "m 옵션은 eye랑 object가 둘다 sky일 때만 가능합니다." << endl;
      }
      break;
    case 'p':
      bool old_picking = g_picking;
      if (g_picking) {
        g_picking = false;
        cout << "picking end by p pressed! g_picking: true -> false" << endl;
      } else {
        g_picking = true;
        if (g_eyeNames[g_currentEyeIdx] == "sky" && g_skyFrameNames[g_currentSkyFrame] == "world-sky") {
          g_currentSkyFrame = (g_currentSkyFrame + 1) % 2;
          cout << "현재 sky frame이 world-sky라서 바꿈: " << g_skyFrameNames[g_currentSkyFrame] << endl;
        }
        cout << "picking start! g_picking: false -> true" << endl;
      }
      break;
  }
  glutPostRedisplay();
}

static void initGlutState(int argc, char *argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);  //  RGBA pixel channels and double buffering
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("CS380: Assignment 2");                       // title the window

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutKeyboardFunc(keyboard);
}

static void initGLState() {
  glClearColor(128. / 255., 200. / 255., 255. / 255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
  if (!g_Gl2Compatible)
    glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initMaterials() {
  // Create some prototype materials
  Material diffuse("./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader");
  Material solid("./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader");

  // copy diffuse prototype and set red color
  g_redDiffuseMat.reset(new Material(diffuse));
  g_redDiffuseMat->getUniforms().put("uColor", Cvec3f(1, 0, 0));

  // copy diffuse prototype and set blue color
  g_blueDiffuseMat.reset(new Material(diffuse));
  g_blueDiffuseMat->getUniforms().put("uColor", Cvec3f(0, 0, 1));

  // normal mapping material
  g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader", "./shaders/normal-gl3.fshader"));
  g_bumpFloorMat->getUniforms().put("uTexColor", shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
  g_bumpFloorMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("FieldstoneNormal.ppm", false)));
  // if the compile errors occur, use below instead (intel mac)
//   g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader", "./shaders/normal-gl3.fshader"));
//   g_bumpFloorMat->getUniforms().put_tex("uTexColor", shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
//   g_bumpFloorMat->getUniforms().put_tex("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("FieldstoneNormal.ppm", false)));

  // copy solid prototype, and set to wireframed rendering
  g_arcballMat.reset(new Material(solid));
  g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
  g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // copy solid prototype, and set to color white
  g_lightMat.reset(new Material(solid));
  g_lightMat->getUniforms().put("uColor", Cvec3f(1, 1, 1));

  // pick shader
  g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"));
}

static void initGeometry() {
  initGround();
  initCubes();
  initSphere();
}

static void constructRobot(shared_ptr<SgTransformNode> base, shared_ptr<Material> material) {

  const double ARM_LEN = 0.7,
    ARM_THICK = 0.25,
    TORSO_LEN = 1.5,
    TORSO_THICK = 0.25,
    TORSO_WIDTH = 1,
    HEAD_RADIUS = 0.25,
    LEG_THICK = 0.2,
    LEG_LEN = 0.7;

  const int NUM_JOINTS = 10,
    NUM_SHAPES = 10;

  struct JointDesc {
      int parent;
      float x, y, z;
  };

  JointDesc jointDesc[NUM_JOINTS] = {
    {-1}, // torso
    {0,  TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper right arm
    {1,  ARM_LEN, 0, 0}, // lower right arm
    {0, -TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper left arm
    {3, -ARM_LEN, 0, 0}, // lower left arm
    {0, TORSO_WIDTH/4, -TORSO_LEN/2, 0}, // upper right leg
    {5, 0, -LEG_LEN, 0}, // lower right leg
    {0, -TORSO_WIDTH/4, -TORSO_LEN/2, 0}, // upper left leg
    {7, 0, -LEG_LEN, 0}, // lower left leg
    {0, 0, TORSO_LEN/2, 0}, // head
  };

  struct ShapeDesc {
      int parentJointId;
      float x, y, z, sx, sy, sz;
      shared_ptr<Geometry> geometry;
  };

  ShapeDesc shapeDesc[NUM_SHAPES] = {
    {0, 0,         0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube}, // torso
    {1, ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // upper right arm
    {2, ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // lower right arm
    {3, -ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // upper left arm
    {4, -ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // lower left arm
    {5, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // upper right leg
    {6, 0, -LEG_LEN/2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube}, // lower left leg
    {7, 0, -LEG_LEN/2,0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // upper right leg
    {8, 0, -LEG_LEN/2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube}, // lower left leg
    {9, 0, HEAD_RADIUS, 0, HEAD_RADIUS, HEAD_RADIUS, HEAD_RADIUS, g_sphere}, // head
  };

  shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

  for (int i = 0; i < NUM_JOINTS; ++i) {
    if (jointDesc[i].parent == -1)
      jointNodes[i] = base;
    else {
      jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
      jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
    }
  }

  for (int i = 0; i < NUM_SHAPES; ++i) {
    shared_ptr<SgGeometryShapeNode> shape(
      new MyShapeNode(shapeDesc[i].geometry,
                      material, // USE MATERIAL as opposed to color
                      Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
                      Cvec3(0, 0, 0),
                      Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
    jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
  }
}

static void initScene() {
  g_world.reset(new SgRootNode());

  g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 4.0))));

  g_groundNode.reset(new SgRbtNode());
  g_groundNode->addChild(shared_ptr<MyShapeNode>(
    new MyShapeNode(g_ground, g_bumpFloorMat, Cvec3(0, g_groundY, 0))));

  g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
  g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));

  constructRobot(g_robot1Node, g_redDiffuseMat); // a Red robot
  constructRobot(g_robot2Node, g_blueDiffuseMat); // a Blue robot

  g_world->addChild(g_skyNode);
  g_world->addChild(g_groundNode);
  g_world->addChild(g_robot1Node);
  g_world->addChild(g_robot2Node);

  cout << "getPathAccumRbt debug 10" << endl;
  g_currentEyeRbt = getPathAccumRbt(g_world, g_skyNode);
}

int main(int argc, char *argv[]) {
  try {
    initGlutState(argc, argv);

    glewInit(); // load the OpenGL extensions

    cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.3") << endl;
    if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
    else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");

    initGLState();
    initMaterials();
    initGeometry();
    initScene();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error &e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}

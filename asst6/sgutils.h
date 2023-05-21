#ifndef SGUTILS_H
#define SGUTILS_H

#include <vector>

#include "scenegraph.h"

struct RbtNodesScanner : public SgNodeVisitor {
  typedef std::vector<std::shared_ptr<SgRbtNode> > SgRbtNodes;

  SgRbtNodes& nodes_;

  RbtNodesScanner(SgRbtNodes& nodes) : nodes_(nodes) {}

  virtual bool visit(SgTransformNode& node) {
    using namespace std;
    shared_ptr<SgRbtNode> rbtPtr = dynamic_pointer_cast<SgRbtNode>(node.shared_from_this());
    if (rbtPtr)
      nodes_.push_back(rbtPtr);
    return true;
  }
};

inline void dumpSgRbtNodes(std::shared_ptr<SgNode> root, std::vector<std::shared_ptr<SgRbtNode> >& rbtNodes) {
  RbtNodesScanner scanner(rbtNodes);
  root->accept(scanner);
}

struct RbtNodesReplacer : public SgNodeVisitor {
    typedef std::vector<RigTForm> RigTForms;

    RigTForms& keyFrame_;
    int index_;

    RbtNodesReplacer(RigTForms& keyFrame, int index)
      : keyFrame_(keyFrame), index_(index) {}

    virtual bool visit(SgTransformNode& node) {
      using namespace std;
//      std::cout << "visit start index: " << index_ << std::endl;
      shared_ptr<SgRbtNode> rbtPtr = dynamic_pointer_cast<SgRbtNode>(node.shared_from_this());
      if (rbtPtr) {
        rbtPtr->setRbt(keyFrame_[index_]);
        ++index_;
//        std::cout << "visit end index: " << index_ << std::endl;
      }

      return true;
    }
};

inline void replaceSgNode(std::shared_ptr<SgNode> root, std::vector<RigTForm>& keyFrame) {
//  std::cout << "keyFrame.size: " << keyFrame.size() << std::endl;
//  std::cout << "index: " << 0 << std::endl;
  RbtNodesReplacer replacer(keyFrame, 0);
  root->accept(replacer);
}


#endif
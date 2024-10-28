//
// Created by wjc on 24-10-28.
//
#include "communication/communication_interface.h"
#include "communication/communicate_ghost.h"
#include <memory>

namespace qcu {
namespace communication {

static std::shared_ptr<Communicator> underlying_communicator = nullptr;

}
}
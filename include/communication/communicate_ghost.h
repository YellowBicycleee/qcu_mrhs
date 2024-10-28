//
// Created by wjc on 24-10-28.
//

#pragma once

namespace qcu {
namespace communication {

template <unsigned int _dims>
struct MemoryGhost {
  void* neighbor_buf[_dims][2]; //2 means fwd bwd
};

}
}
